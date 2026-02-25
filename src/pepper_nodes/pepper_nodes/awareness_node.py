import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from pepper_nodes import PepperNode
from pepper_nodes.utils import Session
import time

class AwarenessNode(PepperNode):

    def __init__(self):
        super().__init__('awareness_node')
        # Gestione connessione (Assumiamo Session gestisca la socket)
        self.session = Session(self.ip, self.port)
        
        try:
            self.awareness_proxy = self.session.get_service("ALBasicAwareness")
            self.motion_proxy = self.session.get_service("ALMotion")
            
            # CONFIGURAZIONE UNA TANTUM ALL'AVVIO
            self.configure_awareness_settings()
            
        except Exception as e:
            self.get_logger().error(f"Errore critico connessione servizi Naoqi: {e}")
            # Qui si potrebbe decidere di terminare il nodo se i servizi non ci sono
        
        # Servizio ROS
        self.srv_tracking = self.create_service(SetBool, 'set_tracking', self.tracking_callback)
        self.get_logger().info("AwarenessNode pronto e configurato.")

    def configure_awareness_settings(self):
        """Configura i parametri di tracking una sola volta all'avvio."""
        try:
            self.get_logger().info("Configurazione parametri ALBasicAwareness...")

            # 1. Reset parametri precedenti per sicurezza
            if self.awareness_proxy.isAwarenessRunning():
                self.awareness_proxy.stopAwareness()

            # 2. Configurazione Stimoli (Solo People)
            self.awareness_proxy.setStimulusDetectionEnabled("Sound", False)
            self.awareness_proxy.setStimulusDetectionEnabled("Movement", False)
            self.awareness_proxy.setStimulusDetectionEnabled("Touch", False)
            self.awareness_proxy.setStimulusDetectionEnabled("People", True)
            
            # 3. Engagement Mode (SemiEngaged = guarda l'utente ma non si gira col corpo)
            # Nota: "SemiEngaged" è ottimo per dialoghi one-to-one
            try:
                self.awareness_proxy.setEngagementMode("SemiEngaged")
            except Exception:
                self.get_logger().warn("SemiEngaged non supportato, uso 'FullyEngaged'")
                self.awareness_proxy.setEngagementMode("FullyEngaged")

            self.life_proxy = self.session.get_service("ALAutonomousLife")
            if self.life_proxy.getState() != "disabled":
                self.get_logger().warn("Disabilito AutonomousLife per permettere il tracking...")
                self.life_proxy.setState("disabled")
            
            # IMPORTANTE: Assicurarsi che i motori della testa abbiano "Stiffness" (rigidità)
            # Se la stiffness è 0, ALBasicAwareness invia comandi ma la testa non si muove.
            self.motion_proxy.setStiffnesses("Head", 1.0)

            # 4. Velocità e fluidità
            # LookStimulusSpeed: 0.0 (lento) a 1.0 (veloce)
            try:
                self.awareness_proxy.setParameter("LookStimulusSpeed", 0.5) 
            except Exception:
                pass

            self.get_logger().info("AWARENESS Configurazione completata con successo.")
            
        except Exception as e:
            self.get_logger().error(f"Errore durante la configurazione iniziale: {e}")

    def shutdown_cleanup(self):
        """Metodo vitale per fermare Pepper alla chiusura del nodo."""
        self.get_logger().warn("Spegnimento in corso: Fermo il tracking...")
        try:
            if self.awareness_proxy.isAwarenessRunning():
                self.awareness_proxy.stopAwareness()
            self.motion_proxy.setStiffnesses("Head", 0.0) # Rilascia i motori per risparmiare energia
        except Exception as e:
            self.get_logger().error(f"Errore durante il cleanup: {e}")

    def tracking_callback(self, request, response):
        if request.data:
            success = self.start_tracking()
            response.message = "Tracking Attivato"
        else:
            success = self.stop_tracking_and_reset()
            response.message = "Tracking Disattivato e Reset"
        
        response.success = success
        return response

    def start_tracking(self):
        """Avvia solo il servizio, i parametri sono già settati."""
        try:
            if self.awareness_proxy.isAwarenessRunning():
                self.get_logger().info("Tracking già attivo.")
                return True

            # IMPORTANTE: Assicurarsi che i motori della testa abbiano "Stiffness" (rigidità)
            # Se la stiffness è 0, ALBasicAwareness invia comandi ma la testa non si muove.
            self.motion_proxy.setStiffnesses("Head", 1.0)
            
            self.awareness_proxy.startAwareness()
            self.get_logger().info("✅ Tracking avviato.")
            return True

        except Exception as e:
            self.get_logger().error(f"Errore start_tracking: {e}")
            return False

    def stop_tracking_and_reset(self):
        """Ferma il tracking e riporta la testa al centro."""
        try:
            # Ferma l'awareness se sta girando
            if self.awareness_proxy.isAwarenessRunning():
                self.awareness_proxy.stopAwareness()

            # Reset posizione testa (Yaw=0, Pitch=-0.15 sguardo orizzontale)
            # Velocità 0.3 è un buon compromesso tra naturalezza e reattività
            self.motion_proxy.setStiffnesses("Head", 1.0)
            self.motion_proxy.setAngles(["HeadYaw", "HeadPitch"], [0.0, -0.15], 0.3)
            
            self.get_logger().info("Tracking fermato, testa resettata.")
            return True
        except Exception as e:
            self.get_logger().error(f"Errore stop_tracking: {e}")
            return False

def main():
    rclpy.init()
    node = AwarenessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()