import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String, Bool
import math
from std_srvs.srv import SetBool
from pathlib import Path # Aggiunto per percorsi generici

# Import messaggi/servizi custom
from pepper_interfaces.srv import Asr
from project_interfaces.msg import SpeechEvent

import os
import numpy as np
import torch

# --- FIX COMPATIBILITÀ TORCHAUDIO ---
# Questo blocco deve stare PRIMA di importare speechbrain
import torchaudio
try:
    torchaudio.list_audio_backends()
except AttributeError:
    # Se la funzione non esiste (torchaudio > 2.1), la creiamo fittizia
    def _list_audio_backends():
        # Restituisce 'soundfile' che è il backend standard usato su Linux
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends
# ------------------------------------
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

# --- CONFIGURAZIONE PERCORSI ---
# Manteniamo la coerenza con l'Orchestrator Node usando la stessa cartella base
HOME_DIR = str(Path.home())
PROJECT_DATA_DIR = os.path.join(HOME_DIR, '.pepper_project_data')
USERS_DB_PATH = os.path.join(PROJECT_DATA_DIR, 'user_database')
MODELS_DIR = os.path.join(PROJECT_DATA_DIR, 'tmp_model') # Spostato qui per pulizia

# Assicurati che esistano
os.makedirs(USERS_DB_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- COSTANTI & STATI ---
SIMILARITY_THRESHOLD = 0.30
DOA_TOLERANCE = 30 

# Stati della FSM
STATE_IDLE = "IDLE"                         # Pronto a ricevere audio
STATE_COMPUTING_ID = "COMPUTING_ID"         # Calcolo embedding (CPU)
STATE_WAITING_ASR = "WAITING_ASR"           # Attesa risposta servizio esterno
STATE_PUBLISHING = "PUBLISHING"             # Pubblicazione risultati

class AudioProcessorNode(Node):
    def __init__(self):
        super().__init__('audio_processor_node')


        # --- STATO INTERNO ---
        self.state = STATE_IDLE
        self.current_doa = 0
        self.current_processing_data = {
            "speaker_id": "unknown",
            "embedding": None,
            "audio_float": None
        }

        # --- MODELLI AI ---
        self.get_logger().info("⏳ Caricamento SpeechBrain...")
        self.spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODELS_DIR # Usa la cartella centralizzata invece di quella locale
        )
        self.user_db = {}
        self.load_user_database()
        self.get_logger().info("✅ Modelli Pronti.")

        # --- SUBSCRIBERS ---
        self.sub_audio = self.create_subscription(
            AudioData, 
            '/speech_audio', 
            self.callback_audio_input, 
            10
        )
        
        self.sub_doa = self.create_subscription(
            Int32,
            'sound_direction',
            self.callback_doa,
            10
        )

        self.pub_event = self.create_publisher(SpeechEvent, '/perception/speech_event', 10)

        # --- CLIENTS & PUBLISHERS ---
        # Nota: Non serve callback_group speciale perché usiamo call_async
        self.asr_client = self.create_client(Asr, 'asr')
        # 4. client tracking
        self.tracking_client = self.create_client(SetBool, 'set_tracking')

        

        self.get_logger().info("🎧 Audio Processor (FSM) Ready.")

    def load_user_database(self):
        if not os.path.exists(USERS_DB_PATH): return
        for user_id in os.listdir(USERS_DB_PATH):
            user_dir = os.path.join(USERS_DB_PATH, user_id)
            if os.path.isdir(user_dir):
                emb_path = os.path.join(user_dir, "voice_embedding.npy")
                if os.path.exists(emb_path):
                    try:
                        self.user_db[user_id] = np.load(emb_path).flatten()
                    except: pass

    # =========================================================================
    #                               CALLBACKS (INGRESSI)
    # =========================================================================

    def set_tracking(self, flag):
        """
        Invia una richiesta al servizio 'set_tracking' con data=False.
        Questo ferma ALBasicAwareness e resetta la testa al centro.
        """
        # 1. Crea la richiesta
        flag_tracking = SetBool.Request()
        flag_tracking.data = flag  
        
        # 2. Controlla che il client sia pronto (opzionale ma consigliato)
        if not self.tracking_client.service_is_ready():
            self.get_logger().warn("Servizio 'set_tracking' non disponibile!")
            return

        # 3. Chiama il servizio in modo ASINCRONO
        # Usiamo call_async per non bloccare il resto dell'orchestrator mentre aspetta
        future = self.tracking_client.call_async(flag_tracking)


    def callback_doa(self, msg):
        self.current_doa = msg.data
        self.current_doa = (self.current_doa+180)%360

    def callback_audio_input(self, msg):
        """
        Punto di ingresso della FSM.
        Accetta l'input SOLO se siamo in IDLE. Altrimenti ignora (drop packet).
        """

        # 1. Controllo Stato: Se sto già lavorando, ignora nuovi input (evita lag)
        if self.state != STATE_IDLE:
            self.get_logger().debug(f"⚠️ Audio ignorato: nodo occupato in stato {self.state}")
            return

        # 2. Controllo DOA
        doa = self.current_doa
        is_frontal = (doa <= DOA_TOLERANCE) or (doa >= 360 - DOA_TOLERANCE)
        if not is_frontal:
            self.get_logger().debug(f"Audio non frontale proveniente da {doa}")
            return # Ignora audio laterale/posteriore

        # 3. Transizione di stato -> COMPUTING_ID
        self.state = STATE_COMPUTING_ID
        self.get_logger().info("🔄 Stato: COMPUTING_ID")
        self.set_tracking(True)

        # Avvia la logica di calcolo (Identity)
        self.step_compute_identity(msg.data)

    # =========================================================================
    #                               LOGICA FSM
    # =========================================================================

    def step_compute_identity(self, raw_bytes):
        """
        Fase 1: Converte audio, calcola embedding, trova ID locale.
        """
        try:
            # -- Elaborazione Pesante (CPU) --
            # In ROS Single Threaded questo blocca brevemente il loop, 
            # ma l'inference di un embedding è veloce (<200ms)
            
            # Conversione Int16 -> Float32
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # SpeechBrain Embedding
            signal_tensor = torch.tensor(audio_float32).unsqueeze(0)
            with torch.no_grad():
                emb_res = self.spk_encoder.encode_batch(signal_tensor)
                new_embedding = emb_res[0,0,:].cpu().numpy().flatten()

            # Confronto DB
            self.load_user_database()
            identified_id = "unknown"
            best_score = -1.0
            for uid, stored_emb in self.user_db.items():
                score = 1 - cosine(new_embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    if score > SIMILARITY_THRESHOLD:
                        identified_id = uid
            
            self.get_logger().info(f"👤 Speaker: {identified_id} (Conf: {best_score:.2f})")

            # Salva i dati temporanei nella classe
            self.current_processing_data["speaker_id"] = identified_id
            self.current_processing_data["embedding"] = new_embedding
            self.current_processing_data["audio_float"] = audio_float32 # Serve per l'ASR

            # Transizione -> ASR
            self.step_request_asr()

        except Exception as e:
            self.get_logger().error(f"Errore in Compute Identity: {e}")
            self.reset_to_idle()

    def step_request_asr(self):
        """
        Fase 2: Chiama il servizio ASR in modo ASINCRONO.
        """
        if not self.asr_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error("ASR Service down!")
            self.reset_to_idle()
            return

        self.state = STATE_WAITING_ASR
        self.get_logger().info("🔄 Stato: WAITING_ASR")

        req = Asr.Request()
        # Recupera l'audio salvato nel passo precedente
        req.audio_features = self.current_processing_data["audio_float"].tolist()

        # Chiamata Asincrona: NON blocca il nodo
        future = self.asr_client.call_async(req)
        # Quando arriva la risposta, esegui callback_asr_done
        future.add_done_callback(self.callback_asr_done)

    def callback_asr_done(self, future):
        """
        Fase 3: Risposta ASR ricevuta.
        """
        try:
            response = future.result()
            transcription = response.text
            
            if transcription:
                self.step_publish_event(transcription)
            else:
                self.get_logger().warn("ASR vuoto.")
                self.reset_to_idle()

        except Exception as e:
            self.get_logger().error(f"ASR Call Failed: {e}")
            self.reset_to_idle()

    def step_publish_event(self, text):
        """
        Fase 4: Pubblica e resetta.
        """
        self.state = STATE_PUBLISHING
        
        msg = SpeechEvent()
        msg.speaker_id = self.current_processing_data["speaker_id"]
        msg.transcription = text
        msg.embedding = self.current_processing_data["embedding"].tolist()

        self.pub_event.publish(msg)
        self.get_logger().info(f"🚀 Evento inviato: '{text}'")
        
        self.reset_to_idle()

    def reset_to_idle(self):
        """Helper per pulire tutto e tornare pronti."""
        self.current_processing_data = {
            "speaker_id": "unknown",
            "embedding": None,
            "audio_float": None
        }
        self.state = STATE_IDLE
        self.get_logger().info("💤 Stato: IDLE (Pronto)")

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessorNode()
    try:
        # Ora possiamo usare lo spin standard Single Threaded!
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()