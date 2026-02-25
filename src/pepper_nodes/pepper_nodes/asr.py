import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
import speech_recognition as sr  # Nuova libreria
from pepper_interfaces.srv import Asr

class LanguageNotSupportedError(Exception):
    def __init__(self, *args):
        super().__init__("Language not supported. Choose one of the following: " + ", ".join(ASRNode.ALLOWED_LANGUAGES))

class ASRNode(Node):

    ALLOWED_LANGUAGES = ['it', 'en']

    def __init__(self):
        super().__init__("asr_node")
        
        # Dichiarazione parametro ROS language
        self.declare_parameter("language", "it", descriptor=ParameterDescriptor(description="Language for audio transcription: 'it' or 'en'"))
        
        # Lettura del parametro
        self.language_param = self.get_parameter("language").get_parameter_value().string_value
        if self.language_param not in ASRNode.ALLOWED_LANGUAGES:
            raise LanguageNotSupportedError()
        
        # Mappatura lingua per Google API
        self.google_lang_code = "it-IT" if self.language_param == "it" else "en-US"

        # Inizializzazione Recognizer (molto leggero rispetto a Whisper)
        self.recognizer = sr.Recognizer()
        
        # Creo service ASR
        self.srv = self.create_service(Asr, "asr", self._callback)
        
        self.get_logger().info(f"ASR Node initialized using Google Web Speech API for language {self.google_lang_code}")

        # === AGGIUNTA: CHIAMATA DI RISCALDAMENTO ===
        self._warm_up()


    def _warm_up(self):
        """
        Invia un breve audio vuoto per inizializzare la connessione con l'API di Google.
        """
        self.get_logger().info("⏳ Warming up Google API connection...")
        try:
            # Creiamo 0.5 secondi di silenzio (array di zeri)
            # 16000 Hz sample rate, int16
            sample_rate = 16000
            duration = 0.5 
            dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
            
            audio_bytes = dummy_audio.tobytes()
            audio_source = sr.AudioData(audio_bytes, sample_rate=sample_rate, sample_width=2)
            
            # Inviamo la richiesta. Ci aspettiamo che fallisca nel riconoscimento 
            # (perché è silenzio), ma avrà aperto il socket di rete.
            self.recognizer.recognize_google(audio_source, language=self.google_lang_code)
            
        except sr.UnknownValueError:
            # Questo è il risultato atteso: "Non ho capito nulla" (perché era silenzio)
            self.get_logger().info("✅ API Warm-up complete (Connection established).")
        except sr.RequestError as e:
            self.get_logger().error(f"❌ Warm-up failed (Network error): {e}")
        except Exception as e:
            # Ignoriamo altri errori in questa fase
            self.get_logger().warn(f"⚠️ Warm-up exception (non-critical): {e}")


            

    def predict(self, audio_data: np.ndarray) -> str:
        """
        Converte l'audio raw (float32) in formato compatibile con Google Speech Recognition
        e invia la richiesta all'API.
        """
        try:
            # 1. Conversione formato: da float32 [-1, 1] a int16 [-32768, 32767]
            # Google SpeechRecognition lavora meglio con interi PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 2. Conversione in bytes
            audio_bytes = audio_int16.tobytes()
            
            # 3. Creazione oggetto AudioData
            # Assumiamo sample_rate=16000 come nel vecchio codice Whisper, 
            # sample_width=2 significa 2 bytes (16 bit) per campione.
            audio_source = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
            
            self.get_logger().info("Sending audio to Google API...")
            
            # 4. Chiamata API (Internet richiesto)
            transcription = self.recognizer.recognize_google(
                audio_source, 
                language=self.google_lang_code
            )
            
            self.get_logger().info(f"ASR Output: {transcription}")
            return transcription

        except sr.UnknownValueError:
            self.get_logger().warn("Google Speech Recognition could not understand audio")
            return "" # Ritorna stringa vuota se non capisce
        except sr.RequestError as e:
            self.get_logger().error(f"Could not request results from Google Speech Recognition service; {e}")
            return "ERROR_API"
        except Exception as e:
            self.get_logger().error(f"Generic Error in ASR: {e}")
            return ""

    def _callback(self, req: Asr.Request, resp: Asr.Response):
        # Conversione input list in array numpy
        audio_list = req.audio_features
        audio_np = np.asarray(audio_list, dtype=np.float32)
        
        # Controllo validità input
        if audio_np.ndim != 1:
            self.get_logger().error("Audio must be a 1D array")
            resp.text = ""
            return resp
            
        text = self.predict(audio_np)
        resp.text = str(text)
        return resp

def main():
    rclpy.init()
    node = ASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()