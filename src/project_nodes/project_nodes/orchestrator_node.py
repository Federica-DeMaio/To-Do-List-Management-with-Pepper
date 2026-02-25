import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32MultiArray
from sensor_msgs.msg import Image
import urllib.parse
import os
import numpy as np
import cv2 # Per il salvataggio delle immagini
from cv_bridge import CvBridge # Da ROS Image a OpenCV
import math
from std_srvs.srv import SetBool
from pathlib import Path # Aggiunto per gestire i percorsi in modo generico e sicuro

# 1. Interfacce del tuo progetto (LLM)
from project_interfaces.srv import LLMInteraction

# 2. Interfacce di Pepper (per TTS e Tablet)
from pepper_interfaces.srv import Text2Speech, LoadUrl

# 3. Interfacce ROS standard e del tuo progetto
from project_interfaces.msg import SpeechEvent


# --- CONFIGURAZIONE PERCORSI ---
# Utilizziamo la cartella Home dell'utente corrente per evitare path assoluti privati
# e risolvere i problemi di os.getcwd() che variano a seconda di dove si lancia il nodo ROS.
HOME_DIR = str(Path.home())
# TODO: Modifica il nome della cartella principale se lo desideri
USERS_DB_PATH = os.path.join(HOME_DIR, '.pepper_project_data', 'user_database') 

# Assicurati che esista
os.makedirs(USERS_DB_PATH, exist_ok=True)


# --- DEFINIZIONE STATI ---
STATE_WAITING = "WAITING_FOR_USER"  # Nessuno davanti
STATE_WAITING_FOR_GREETING = "WAIT_GREETING" # Attesa utente saluta
STATE_LISTENING = "LISTENING"       # In attesa di input vocale
STATE_PROCESSING = "PROCESSING"     # LLM sta pensando
STATE_SPEAKING = "SPEAKING"         # Il robot sta parlando

STATE_REG_WAIT_TTS = "REG_WAITING_TTS" # Quando il robot chiede il nome nella fase di registrazione
STATE_REG_WAIT_AUDIO = "REG_WAITING_AUDIO" # Aspetto che il nodo audio dia responso
STATE_REG_SAVING = "REG_SAVING" # Stato per il salvataggio dei dati

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        # --- STATO INTERNO ---
        self.state = STATE_WAITING
        self.current_user_id = None
        self.bridge = CvBridge()
        # CACHE: Memorizza l'ultima immagine sconosciuta per il salvataggio
        self.last_unknown_face_data = None
        self.last_unknown_voice_data = None
        self.last_unknown_name_data = None

        self.time_for_name = None

        # --- SUBSCRIBERS (INPUT) ---
        # Riceve l'ID utente (dal Face Recognition o simulatore)
        self.create_subscription(
            String, '/perception/user_id', self.callback_user_id, 10
        )
        # Riceve l'immagine dello sconosciuto (dal Face Recognition)
        self.create_subscription(
            Image, '/perception/face_to_register', self.callback_face_to_register, 1
        )
        
        # 3. SPEECH EVENT (Tutto l'audio passa di qui: Comandi e Nomi per registrazione)
        self.sub_speech = self.create_subscription(
            SpeechEvent, '/perception/speech_event', self.callback_user_input, 10
        )

        # Disattivazione microfono per ignorare voce Pepper
        self.pub_is_speaking = self.create_publisher(
            Bool, 
            '/pepper/is_speaking', 
            10
        )
        
        # --- CLIENTS (OUTPUT & LOGIC) ---
        
        # 1. Client LLM (Il tuo nodo memoria)
        self.llm_client = self.create_client(LLMInteraction, 'llm_interaction_service')
        
        # 2. Client TTS (Il nodo Text2SpeechNode fornito)
        self.tts_client = self.create_client(Text2Speech, 'tts')
        
        # 3. Client Tablet (Il nodo TabletNode fornito)
        self.tablet_client = self.create_client(LoadUrl, 'load_url')

        # 4. Client tracking
        self.tracking_client = self.create_client(SetBool, 'set_tracking')

        # Timer per debug
        self.create_timer(2.0, self.log_status)
        self.get_logger().info("🤖 Orchestrator Ready. Waiting for users...")

        self.set_tracking(False)

    def log_status(self):
        self.get_logger().info(f"[STATE: {self.state}] User: {self.current_user_id}")

    # =========================================================================
    #                               CALLBACK DI INPUT
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


    def reset_to_waiting(self):
        self.state = STATE_WAITING
        self.current_user_id = None
        if hasattr(self, 'time_for_name') and self.time_for_name:
            self.time_for_name.cancel()
        self.update_tablet([]) # Pulisci tablet
        self.set_tracking(False)

    def callback_face_to_register(self, msg):
        """
        Callback per ricevere l'immagine di un volto sconosciuto. 
        Mette in cache il dato per l'uso futuro nel salvataggio.
        """
        # Conserviamo l'immagine solo se siamo in uno stato di attesa o registrazione
        # Accettandola in questi stati, permetto di fare più acquisizioni durante la fase di registrazione. 
        # Ogni volta sovrascrivo la stessa cache ma ho più probabilità di fare un'acquisizione nitida
        if self.state in [STATE_WAITING, STATE_WAITING_FOR_GREETING, STATE_REG_WAIT_TTS, STATE_REG_WAIT_AUDIO]:
            self.last_unknown_face_data = msg
            self.get_logger().debug("🖼️ Immagine sconosciuta ricevuta e salvata in cache.")
            
    
    def callback_user_id(self, msg):
        """
        Gestisce il rilevamento presenza utente con logica di stabilizzazione:
        - Priorità all'ID numerico rispetto a "unknown".
        - Ignora "unknown" se l'utente è già identificato.
        - Resetta se cambia l'ID numerico.
        """
        detected_user = msg.data.strip()

        # 1. CASO: NESSUNO DAVANTI
        # Se non c'è nessuno o stringa vuota/none
        if not detected_user or detected_user.lower() == "none":
            if self.state != STATE_WAITING:
                self.get_logger().info("👋 Utente perso. Reset.")
                self.reset_to_waiting()
            return
        
        # 2. CASO: PRIMA RILEVAZIONE (Eravamo in Waiting)
        if self.state == STATE_WAITING:
            self.get_logger().info(f"👀 Utente rilevato ({detected_user}). Attendo saluto...")
            self.current_user_id = detected_user
            self.state = STATE_WAITING_FOR_GREETING
            #self.set_tracking(False)
            return

        # 3. GESTIONE CAMBI DI STATO (Mentre l'utente è davanti)
        if self.current_user_id is None:
            # Should not happen se non siamo in WAITING, ma per sicurezza:
            self.current_user_id = detected_user
            return

        # --- LOGICA DI FILTRO E STABILIZZAZIONE ---

        # A. UPGRADE: Da Unknown -> a ID Noto
        # (Es. La camera ha messo a fuoco meglio e ora sa chi è)
        if self.current_user_id == "unknown" and detected_user.isdigit():
            self.get_logger().info(f"✅ Identificazione riuscita: unknown -> {detected_user}")
            self.current_user_id = detected_user
            
            # Se stavo provando a registrarlo, INTERROMPO la registrazione
            if self.state in [STATE_REG_WAIT_TTS, STATE_REG_WAIT_AUDIO, STATE_REG_SAVING]:
                self.get_logger().warn(f"🛑 Registrazione interrotta: L'utente è in realtà ID {detected_user}")
                # Reset timer se attivo
                if hasattr(self, 'time_for_name') and self.time_for_name:
                    self.time_for_name.cancel()
                
                # Torno in attesa di saluto o comando, trattandolo come utente noto
                self.state = STATE_WAITING_FOR_GREETING
                self.set_tracking(False)
                # Opzionale: update tablet vuoto per pulire eventuali stati precedenti
                self.update_tablet([])
                self.send_speech_request("Interrompo la registrazione, perché ti ho riconosciuto. Resetto la sessione")

        # B. NOISE FILTER: Da ID Noto -> a Unknown
        # (Es. L'utente ha girato la testa e il riconoscimento ha fallito per un frame)
        elif self.current_user_id.isdigit() and detected_user == "unknown":
            # MANTENIAMO IL DIGIT. Ignoriamo l'unknown momentaneo.
            # self.get_logger().debug("🛡️ Ignoro 'unknown' temporaneo su utente già identificato.")
            pass

        # C. SWITCH UTENTE: Da ID A -> a ID B (o da Digit -> Digit diverso)
        # (Es. Una persona ha spinto via l'altra)
        elif detected_user != self.current_user_id:
            # Nota: qui entra anche se passa da '5' a '7', o da '5' a 'unknown' (ma il caso unknown è gestito sopra dall'elif B)
            # Quindi qui gestiamo effettivamente cambio di ID o cambio ID -> Unknown (se logica B non esistesse)
            
            self.get_logger().warn(f"🔄 Cambio utente rilevato: {self.current_user_id} -> {detected_user}")
            self.current_user_id = detected_user
            
            # Reset logico come richiesto: si torna in attesa di saluto
            self.state = STATE_WAITING_FOR_GREETING
            self.set_tracking(False)

            
            # Pulizia timer registrazione se attivi
            if hasattr(self, 'time_for_name') and self.time_for_name:
                self.time_for_name.cancel()
            
            self.update_tablet([]) # Pulisci tablet per il nuovo utente
            self.send_speech_request("Ho rilevato un cambio di utente. Resetto la sessione.")


    def callback_user_input(self, msg):
        """
        Unica callback audio. Decide come usare l'audio (Comando o Registrazione).
        """
        voice_id = msg.speaker_id
        user_text = msg.transcription.strip()
        embedding = msg.embedding
        
        if not user_text: return

        # --- FILTRI STATO ---
        if self.state in [STATE_PROCESSING, STATE_SPEAKING, STATE_REG_WAIT_TTS, STATE_REG_SAVING]:
            self.get_logger().warn(f"Ignoro input '{user_text}' nello stato {self.state}")
            return

        if self.state == STATE_WAITING:
            self.get_logger().warn("Sento voci ma non vedo nessuno (o face recognition è lento).")
            return

        self.set_tracking(True)


        if self.state == STATE_WAITING_FOR_GREETING:
            self.get_logger().info(f"👂 Saluto ricevuto: '{user_text}' da ID Visivo: {self.current_user_id}")
            
            # Caso A: Utente SCONOSCIUTO (Visivamente)
            if self.current_user_id == "unknown":
                # Avviamo la registrazione solo se l'immagine è già arrivata in cache
                if self.last_unknown_face_data:
                    self.get_logger().info("👀 Utente sconosciuto rilevato! Inizio registrazione...")
                    self.start_registration_sequence()
                else:
                    self.get_logger().warn("Utente sconosciuto rilevato, ma attendo l'immagine (topic face_to_register).")
                    self.state = STATE_WAITING_FOR_GREETING
                    self.send_speech_request("Non ti vedo bene avvicinati un po'.")
                    return
            
            # Caso B: Utente CONOSCIUTO (Visivamente)
            elif self.current_user_id and self.current_user_id.isdigit():
                # Passo il saluto all'LLM che risponderà "Ciao [Nome]..."
                self.call_llm_service(user_text)

        # Se siamo in LISTENING, l'utente sta gestendo la sua to do list
        if self.state == STATE_LISTENING:

            # Sicurezza: Match ID Faccia == ID Voce
            if voice_id == self.current_user_id:
                self.get_logger().info(f"👂 Input: '{user_text}' -> Chiamo LLM")
                self.state = STATE_PROCESSING # Blocco input
                self.call_llm_service(user_text)
            
            else:
                self.get_logger().warn(f"⛔ Conflitto Identità: Vedo {self.current_user_id}, Sento {voice_id}.")
                self.state = STATE_SPEAKING
                self.send_speech_request("La tua voce non corrisponde, ripeti il comando")
                return

        elif self.state == STATE_REG_WAIT_AUDIO:
            # Cancello timer che dà il tempo all'utente per dire il nome.
            if hasattr(self, 'time_for_name') and self.time_for_name:
                self.time_for_name.cancel()
            if voice_id == "unknown" and self.current_user_id == "unknown": # Se sto avendo a che fare per la prima volta con un nuovo utente.
                self.get_logger().warn("⛔ Il nuovo utente ha parlato")
                self.last_unknown_voice_data = embedding
                self.state = STATE_PROCESSING # Blocco input
                self.call_llm_service(user_text) # per estrazione del nome
            else:
                # La registrazione non va a buon fine, il nuovo utente ha una voce uguale all'utente già registrato
                self.get_logger().warn(f"⛔ Conflitto Identità: Vedo un nuovo utente, Sento {voice_id}.")
                self.state=STATE_WAITING_FOR_GREETING
                self.send_speech_request("La tua voce è già registrata per un altro utente. Resetto la sessione")
                self.set_tracking(False)
                return


    # =========================================================================
    #                               SEQUENZA REGISTRAZIONE
    # =========================================================================
    
    def start_registration_sequence(self):
        """Avvia la sequenza di registrazione: Chiedi il nome (TTS)."""
        # Prevenzione di re-entry
        if self.state != STATE_WAITING_FOR_GREETING: return

        self.state = STATE_REG_WAIT_TTS
        self.send_speech_request("Ciao, non ti riconosco. Come ti chiami?")

    
    def save_registration_data(self):
        """Salva tutti i dati raccolti (Immagine + Nome + Embedding) dalla cache."""
        
        if self.last_unknown_face_data is None or self.last_unknown_voice_data is None or self.last_unknown_name_data is None:
            self.get_logger().error("Tentativo di salvataggio senza immagine in cache. Fallimento!")
            self.state=STATE_WAITING_FOR_GREETING
            self.set_tracking(False)
            self.send_speech_request("Ops, la registrazione non è andata a buon fine. Ripetere l'operazione da capo.")
            return

        try:
            # 1. Trova un ID univoco
            all_users = [d for d in os.listdir(USERS_DB_PATH) if os.path.isdir(os.path.join(USERS_DB_PATH, d)) and d.isdigit()]
            new_id = len(all_users) + 1
            user_folder = os.path.join(USERS_DB_PATH, str(new_id))
            os.makedirs(user_folder, exist_ok=True)
            
            # 2. Salva Immagine (dalla cache del topic)
            image_msg = self.last_unknown_face_data 
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            face_path = os.path.join(user_folder, "face.png")
            cv2.imwrite(face_path, cv_image)
            
            # 3. Salva Nome
            name = self.last_unknown_name_data
            with open(os.path.join(user_folder, "name.txt"), 'w') as f:
                f.write(name)

            # 4. Salva Embedding Vocale
            embedding_path = os.path.join(user_folder, "voice_embedding.npy")
            np_embedding = np.array(self.last_unknown_voice_data, dtype=np.float32) 
            np.save(embedding_path, np_embedding)
            
            self.get_logger().info(f"💾 Utente '{name}' registrato con ID: {new_id} in {user_folder}.")

            
            # Reset cache e variabili di stato
            self.last_unknown_face_data = None
            self.last_unknown_voice_data = None
            self.last_unknown_name_data = None
            self.current_user_id = str(new_id)
            self.state = STATE_LISTENING
            
            # Final TTS
            #self.send_speech_request(f"Perfetto {name}. Ora dimmi cosa posso fare per te.")

        except Exception as e:
            self.get_logger().error(f"Errore durante il salvataggio dei dati: {e}")
            self.state = STATE_WAITING # Fallimento


    # =========================================================================
    #                               LOGICA LLM
    # =========================================================================

    def call_llm_service(self, text):
        if not self.llm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("LLM Service non disponibile!")
            self.state = STATE_LISTENING
            return
        
        req = LLMInteraction.Request()
        req.user_input = text
        req.user_id = self.current_user_id if self.current_user_id else "unknown" # è unknown in fase di registrazione, oppure un id numerico in fase di utilizzo
        if self.current_user_id and self.current_user_id.isdigit():
            # Costruiamo il percorso completo al file name.txt
            name_file_path = os.path.join(USERS_DB_PATH, str(self.current_user_id), "name.txt")
            
            # Controlliamo se il file esiste PRIMA di aprirlo
            if os.path.exists(name_file_path):
                try:
                    with open(name_file_path, 'r') as f:
                        # .strip() rimuove eventuali spazi o \n alla fine
                        req.user_name = f.read().strip() 
                except Exception as e:
                    self.get_logger().error(f"Impossibile leggere il nome utente: {e}")
            else:
                # Se il file non esiste, manteniamo "unknown" o usiamo l'ID come fallback se preferisci
                pass
        if self.state == STATE_WAITING_FOR_GREETING:
            req.start = True
        elif self.state == STATE_PROCESSING:
            req.start = False
        

        future = self.llm_client.call_async(req)
        future.add_done_callback(self.handle_llm_response)

    def handle_llm_response(self, future):
        try:
            response = future.result()

            text_to_say = response.text_response
            operation = response.operation
            content = response.item_content

            self.get_logger().info(f"🧠 Risposta: {text_to_say} | Op: {operation}")

            # Lista operazioni che aggiornano il tablet
            # Includiamo sia 'multiple_ops' (continua) che 'multiple_ops_exit' (chiude)
            ui_update_ops = ["add_task", "remove_task", "show_list", "empty_list",
                             "greeting_user", "multiple_ops", "multiple_ops_exit"]

            if operation in ui_update_ops:
                self.update_tablet(content)

                # --- LOGICA DI CHIUSURA TRACKING ---
                # Spegniamo il tracking se è un saluto puro O se è un'operazione multipla che include un saluto
                should_stop_tracking = (operation == "greeting_user") or (operation == "multiple_ops_exit")

                if should_stop_tracking and self.state == STATE_PROCESSING:
                    self.get_logger().info("👋 Saluto rilevato: Disattivazione tracking.")
                    self.set_tracking(False)

                self.state = STATE_SPEAKING

            elif operation == "identification":
                if content and len(content) > 0:
                    name = content[0]
                    self.update_tablet([name])
                    self.last_unknown_name_data = name
                    self.state = STATE_REG_SAVING
                else:
                    self.get_logger().warn("LLM IDENTITY con lista vuota.")
                    text_to_say = "Scusa, non ho capito il nome. Puoi ripetere?"
                    self.state = STATE_REG_WAIT_TTS
            else:
                self.state = STATE_SPEAKING

            self.send_speech_request(text_to_say)

        except Exception as e:
            self.get_logger().error(f"LLM Error: {e}")
            self.state = STATE_LISTENING

    # =========================================================================
    #                               INTERAZIONE ROBOT
    # =========================================================================

    def send_speech_request(self, text):
        """Chiama il servizio Text2Speech"""
        if not self.tts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("TTS Service non disponibile!")
            self.state = STATE_LISTENING
            return
        
        self.pub_is_speaking.publish(Bool(data=True))
        self.get_logger().debug("🔇 Muto attivato (Robot parla)")

        req = Text2Speech.Request()
        req.speech = text
        req.volume= 70 # Volume standard
        req.language = Text2Speech.Request.ITALIAN # O ENGLISH
        
        future = self.tts_client.call_async(req)
        # Importante: Quando il robot finisce di parlare, chiamiamo on_speech_done
        future.add_done_callback(self.on_speech_done)

    def on_speech_done(self, future):
        """Callback eseguita quando il TTS ha finito"""
        try:
            res = future.result() 
            self.get_logger().debug(f"TTS Finito: {res.ack}")
        except Exception as e:
            self.get_logger().error(f"TTS Failed: {e}")
        
        # 2. AVVISO AUDIO: RIATTIVA L'ASCOLTO
        self.pub_is_speaking.publish(Bool(data=False))
        self.get_logger().debug("👂 Muto disattivato (Robot ha finito)")

        # Logica di transizione
        if self.state == STATE_SPEAKING:
            # Flusso di dialogo normale
            self.get_logger().info("🎤 Torno in ascolto.")
            self.state = STATE_LISTENING
        
        elif self.state == STATE_REG_WAIT_TTS:
            # Flusso di registrazione: Il robot ha finito di chiedere il nome.
            self.get_logger().info("🎤 Robot ha chiesto il nome. Attendo che l'utente parli.")
            self.state = STATE_REG_WAIT_AUDIO

            # Se l'utente parlerà scatterà la callback user_input
            # Timer per dare tempo all'utente di rispondere
            self.time_for_name = self.create_timer(15.0, self.repeat_name)


        elif self.state == STATE_REG_SAVING:
            # Il nome dell'utente è stato ottenuto, posso memorizzarlo
            self.save_registration_data()
        
        elif self.state == STATE_WAITING_FOR_GREETING:
            pass
        
        else:
            self.get_logger().warn(f"TTS finito in stato inatteso: {self.state}")
            self.state = STATE_WAITING # Fail-safe

    def repeat_name(self):
        self.get_logger().warn("Nome o Impronta vocale non rilevata. Ritorno a chiedere il nome.")
        self.state = STATE_REG_WAIT_TTS 
        self.time_for_name.cancel() # Riazzerro timer
        self.send_speech_request("Non ho capito. Mi ripeti il tuo nome?")
        return
        
    
    def update_tablet(self, todo_list):
        """Genera HTML e chiama il servizio LoadUrl"""
        if not self.tablet_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Tablet Service non disponibile.")
            return

        # 1. Costruisci HTML
        # Usiamo CSS inline per farlo carino su Pepper
        html = """
        <html>
        <body style='background-color:#f0f0f0; font-family:Arial; text-align:center; padding:20px;'>
            <h1 style='color:#e63946;'>To-Do List</h1>
            <ul style='list-style-type:none; padding:0;'>
        """
        if not todo_list:
            html += "<li><i>(Lista vuota)</i></li>"
        else:
            for item in todo_list:
                html += f"<li style='background:#fff; margin:10px; padding:15px; border-radius:10px; font-size:24px;'>{item}</li>"
        
        html += "</ul></body></html>"

        # 2. Converti in Data URI (per non usare file esterni)
        # Formato: data:text/html;charset=utf-8,CONTENUTO_CODIFICATO
        encoded_html = urllib.parse.quote(html)
        data_uri = f"data:text/html;charset=utf-8,{encoded_html}"

        # 3. Chiama Servizio
        req = LoadUrl.Request()
        req.url = data_uri
        
        self.tablet_client.call_async(req)
        self.get_logger().info("📱 Tablet aggiornato.")

def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()