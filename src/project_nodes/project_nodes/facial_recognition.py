import os
import rclpy
from rclpy.node import Node
from deepface import DeepFace
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import threading
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI ---
# Manteniamo la coerenza con gli altri nodi usando la stessa cartella base
HOME_DIR = str(Path.home())
PROJECT_DATA_DIR = os.path.join(HOME_DIR, '.pepper_project_data')
USERS_DB_PATH = os.path.join(PROJECT_DATA_DIR, 'user_database')

# Creiamo le cartelle se non esistono
os.makedirs(USERS_DB_PATH, exist_ok=True)

# Diciamo a DeepFace di salvare i suoi modelli pesanti all'interno della cartella di progetto
os.environ["DEEPFACE_HOME"] = PROJECT_DATA_DIR

# --- COSTANTI ---
PADDING = 25
DETECTION_THRESHOLD = 0.4
# Non usiamo più FPS per contare i frame, ma un timer in secondi
PROCESS_TIMER_PERIOD = 0.5  # Processa un frame ogni 0.5 secondi (2 FPS)
MAX_BEFORE_FORGETTING = 1
STATE_WAITING = 'WAITING_FOR_USER'
STATE_IDENTIFYING = 'IDENTIFYING_USER'
MODEL_NAME = 'ArcFace'

class FacialRecognition(Node):

    def __init__(self):
        super().__init__('facial_recognition')
        self.to_forget = 0
        self.identified_user = None
        self.state = STATE_WAITING
        self.bridge = CvBridge()
        
        # Variabile per conservare l'ultima immagine ricevuta
        self.latest_image_msg = None
        # Lock per evitare race condition (anche se con SingleThreadedExecutor non è strettamente necessario, è good practice)
        self.img_lock = threading.Lock()

        # QoS profile di default (KeepLast, Depth 1) va bene
        self.image_subscription = self.create_subscription(
            Image, 
            'in_rgb', 
            self.image_callback, 
            1
        )
        self.user_id_publisher = self.create_publisher(String, '/perception/user_id', 10)
        self.face_publisher = self.create_publisher(Image, '/perception/face_to_register', 10)

        # Usiamo il percorso centralizzato configurato in alto
        self.images_folder = USERS_DB_PATH

        self.get_logger().info(f"⏳ Caricamento modello {MODEL_NAME} in {PROJECT_DATA_DIR}...")
        try:
            DeepFace.build_model(MODEL_NAME)
            self.get_logger().info("✅ Modello caricato in memoria.")
        except Exception as e:
            self.get_logger().error(f"Errore caricamento modello: {e}")

        # Timer per l'elaborazione (Sostituisce il conteggio manuale dei frame)
        self.create_timer(PROCESS_TIMER_PERIOD, self.process_face_callback)
        self.create_timer(5.0, self.log_status)
        
        self.get_logger().info("Facial Recognition node ready. Processing asynchronously.")

    def log_status(self):
        self.get_logger().info(f"[STATE: {self.state}]")

    def image_callback(self, msg):
        """
        Questa callback deve essere velocissima. 
        Si limita a salvare l'ultimo dato arrivato.
        """
        with self.img_lock:
            self.latest_image_msg = msg

    def process_face_callback(self):
        """
        Viene chiamata dal Timer. Prende l'ultima foto disponibile e la elabora.
        Se l'elaborazione è lenta, il prossimo timer scatterà in ritardo (in SingleThreadedExecutor),
        ma prenderà SEMPRE l'immagine più nuova, senza code artificiali.
        """
        # 1. Recupera l'immagine in modo thread-safe
        msg = None
        with self.img_lock:
            if self.latest_image_msg is None:
                return # Nessuna immagine arrivata ancora
            msg = self.latest_image_msg
            # Opzionale: self.latest_image_msg = None # Se vuoi evitare di riprocessare la stessa img
        
        # 2. Logica di Rilevamento (Identica alla tua, ma pulita dal conteggio frame)
        face_data = self.detect_faces(msg)

        if self.state == STATE_WAITING:
            if face_data:
                self.to_forget = 0
                self.state = STATE_IDENTIFYING
                self.identify_user(msg, face_data)
            else:
                self.get_logger().debug('Nessun volto, in attesa...')
                self.identified_user = None
                self.user_id_publisher.publish(String(data="none"))
                pass # Nessuno

        elif self.state == STATE_IDENTIFYING:
            if face_data:
                self.to_forget = 0
                self.identify_user(msg, face_data)
            else:
                self.to_forget += 1
                if self.to_forget >= MAX_BEFORE_FORGETTING:
                    self.get_logger().info('👋 Nessun volto. Reset stato.')
                    self.identified_user = None
                    self.user_id_publisher.publish(String(data="none"))
                    self.state = STATE_WAITING

    def detect_faces(self, image: Image) -> dict | None:
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        try:
            # Backend detectors: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'centerface'
            # yolov11n è molto veloce, ottima scelta.
            results = DeepFace.extract_faces(cv_image, enforce_detection=True, detector_backend='yolov11n')
            if len(results) == 1:
                return results[0]
            else:
                return None
        except ValueError:
            return None

    def identify_user(self, full_image_msg: Image, face_data: dict):
        cv_full_image = self.bridge.imgmsg_to_cv2(full_image_msg, desired_encoding='bgr8')
        area = face_data['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']
        
        h_img, w_img, _ = cv_full_image.shape
        y1 = max(0, y - PADDING)
        y2 = min(h_img, y + h + PADDING)
        x1 = max(0, x - PADDING)
        x2 = min(w_img, x + w + PADDING)
        
        cv_face_cropped = cv_full_image[y1:y2, x1:x2]

        found_id = None
        
        if os.path.exists(self.images_folder):
            user_folders = [f for f in os.listdir(self.images_folder) if f.isdigit()]
            
            for user_id in user_folders:
                user_img_path = os.path.join(self.images_folder, user_id, "face.png")
                
                if not os.path.exists(user_img_path):
                    continue

                try:
                    result = DeepFace.verify(
                        img1_path=cv_face_cropped, 
                        img2_path=user_img_path, 
                        enforce_detection=False, 
                        threshold=DETECTION_THRESHOLD,
                        model_name=MODEL_NAME
                    )
                    
                    if result['verified']:
                        found_id = user_id
                        self.get_logger().info(f"✅ Riconosciuto ID: {found_id} con confidenza {result['distance']:.2f}")
                        break
                except Exception as e:
                    self.get_logger().warn(f"Errore DeepFace: {e}")

        if found_id:
            self.identified_user = found_id
            self.user_id_publisher.publish(String(data=found_id))
        else:
            if self.identified_user != "unknown":
                self.get_logger().info("❓ Utente Sconosciuto.")
            
            self.identified_user = "unknown"
            self.user_id_publisher.publish(String(data="unknown"))
            
            try:
                face_msg = self.bridge.cv2_to_imgmsg(cv_face_cropped, encoding="bgr8")
                self.face_publisher.publish(face_msg)
            except Exception as e:
                self.get_logger().error(f"Errore pub faccia: {e}")

def main():
    rclpy.init()
    facial_recognition = FacialRecognition()
    # Usa MultiThreadedExecutor se vuoi che la ricezione immagini continui mentre processa
    # Ma per questo caso specifico (sovrascrittura ultima immagine), lo spin base va bene
    # perché il Timer e la Subscription sono serializzati nello stesso thread.
    rclpy.spin(facial_recognition)
    rclpy.shutdown()

if __name__ == '__main__':
    main()