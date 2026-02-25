from ultralytics import YOLO
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from pepper_interfaces.srv import ObjectDetection
from pepper_interfaces.msg import Detection, Detections
import cv2
from cv_bridge import CvBridge
from typing import List


class ObjDetector(Node):
    NODE_NAME  = "object_detector_node"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        self.model = YOLO("yolo11n") #Model can be changed to other YOLOv8 models like "yolov8n", "yolov8s", etc. https://docs.ultralytics.com/it/models/yolo11/#performance-metrics
        self.br = CvBridge()
        
        self.pub_detections = self.create_publisher(Detections, "/in_rgb/detect", qos_profile=10) #Detections contiene un array di Detecntion, cioè un vettore di coordinate float, una classe ID e una stringa. Per ogni immagine ci sono più rilevazioni, ciascuna di essa avrà bounding box, classe e stringa associata. qos_profile indica dimensione della coda. Se il subscriber è lento o si disconnette, il publisher gli rinvia solo le ultime 10 pubblicazioni, le altre vengono sovrascritte.
        
        self.pub_image = self.create_publisher(Image, "/in_rgb/view", qos_profile=10) #Pubblicazione immagine comprensiva dei bounding box su ogni oggetto rilevato.
        
        self.detect_srv = self.create_service(ObjectDetection, "detect_objects", self.detect_callback, callback_group=MutuallyExclusiveCallbackGroup()) #il service offre lo stesso servizio di subscriber publisher ma permette di farlo su richiesta, chiedendo in specifici momenti di eseguire la rilevazione. ObjectDetection.srv definisce una Request che contiene una immagine acquisita Image  e la response che contiene un vettore di Detenction --> parametri bounding box, classe e stringa.
        
        self.sub_image = self.create_subscription(Image, "/in_rgb", self.detect, qos_profile=1, callback_group=MutuallyExclusiveCallbackGroup()) #il nodo legge dal topic in cui la camera di Pepper pubblica le acquisizioni
        
        self.get_logger().info("Object Detector Node has been started.")
    
    def detect(self, img_msg: Image): # callback invocata quando subscriber legge immagine pubblicata dalla camera di Pepper sul topic /in_rgb 
        img = self.br.imgmsg_to_cv2(img_msg)   #col bridge cabia struttura dati dell'immagine da messaggio ROS a array numpy usato da OpenCV.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # conversione colori BGR usata da OpenCV a RGB
        predict = self.model(img)[0] #il modello può lavorare con batch e restituisce una lista di predizioni. In questo caso lavoriamo con singola immagine, estraiamo dalla posizione 0 del batch.
        detections = self._predict2detect_msg(predict) #preparazione dei messaggi per la pubblicazione sul topic /in_rgb/detect
        msg = Detections(detections=detections)  #creazione dell'array di detection
        self.pub_detections.publish(msg) #pubblicazione sul topic /in_rgb/detect
        nw_img = self._draw_boxes(img, predict) #disegno dei bounding box sulla nuova immagine
        nw_img_msg = self.br.cv2_to_imgmsg(nw_img, encoding='rgb8') #conversione da numpy a messaggio ROS
        self.pub_image.publish(nw_img_msg) #pubblicazione immagine coi bounding box sul topic /in_rgb/view
        return detections
    
    def detect_callback(self, request: ObjectDetection.Request, response: ObjectDetection.Response):
        img_msg = request.image #prendo img dalla richiesta
        detections = self.detect(img_msg) #invoco detect anche per la pubblicazione visiva sul topic
        response.detections = detections #caricamento risposta
        return response #invio risposta.
    
    def _predict2detect_msg(self, predict) -> List[Detection]:
        boxes = predict.boxes.xyxy.tolist()  #dalle predizioni ottiene tutti gli angoli dei bounding box per ogni bounding box rilevato.
        names = predict.names # stringhe per ogni bb
        class_ids = predict.boxes.cls.tolist() # classi per ogni bb
        
        assert len(boxes) == len(class_ids), "Mismatch between boxes and class IDs"
        class_ids = list(map(int, class_ids))
        detetions = []
        for i, box in enumerate(boxes):
            detetion = Detection()
            detetion.class_id = class_ids[i]
            detetion.class_name = names[class_ids[i]]
            detetion.box = box
            detetions.append(detetion)
        return detetions

    def _draw_boxes(self, org_img, predict):

        boxes = predict.boxes.xyxy.tolist()
        names = predict.names
        class_ids = predict.boxes.cls.tolist()
        nw_img = org_img

        # Draw bounding boxes on the image
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = names[class_id]
            nw_img = cv2.rectangle(nw_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            # Put class name
            cv2.putText(nw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=2)
        
        return nw_img

def test():
    from threading import Thread
    import sys
    from pathlib import Path
    curr_dir = Path(__file__).parent
    rclpy.init()
    node = ObjDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    th = Thread(target=executor.spin, daemon=True)
    th.start()
    image = cv2.imread(curr_dir.joinpath("bus.jpg").as_posix())
    assert image is not None, "Image not found"
    image_msg = node.br.cv2_to_imgmsg(image)
    client = node.create_client(ObjectDetection, "detect_objects")
    req = ObjectDetection.Request()
    req.image = image_msg
    resp = client.call(req)
    print(resp)
    sys.exit()


def main():
    rclpy.init()
    node = ObjDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()

if __name__ == "__main__":
    test()
