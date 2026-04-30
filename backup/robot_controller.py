import pybullet as p
import time

class TeachAndPlayController:
    def __init__(self, robot_id, end_effector_index):
        """
        Inicjalizuje kontroler robota.
        :param robot_id: ID robota w symulacji PyBullet
        :param end_effector_index: Indeks przegubu, który jest "końcówką" robota
        """
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.waypoints = []  # Lista przechowująca zapisane punkty (współrzędne X, Y, Z)
        self.is_playing = False  # Czy obecnie odtwarzamy sekwencję

    def save_waypoint(self, target_pos):
        """Zapisuje aktualną pozycję celu do listy."""
        # Kopiujemy listę, żeby uniknąć problemów z referencjami w Pythonie
        self.waypoints.append(list(target_pos))
        print(f"[{len(self.waypoints)}] Zapisano punkt: {target_pos}")

    def clear_sequence(self):
        """Czyści pamięć ruchów."""
        self.waypoints.clear()
        print("Pamięć sekwencji została wyczyszczona.")

    def play_sequence(self, target_marker_id):
        """
        Odtwarza zapisaną sekwencję ruchów.
        :param target_marker_id: ID czerwonej kulki, żebyśmy widzieli za czym podąża robot
        """
        if not self.waypoints:
            print("Brak zapisanych punktów do odtworzenia!")
            return

        print("▶ Rozpoczynam odtwarzanie sekwencji...")
        self.is_playing = True
        
        for pos in self.waypoints:
            print(f"Jadę do punktu: {pos}")
            
            # Przesuwamy wizualny znacznik (czerwoną kulkę) na zapisany punkt
            p.resetBasePositionAndOrientation(target_marker_id, pos, [0, 0, 0, 1])
            
            # Obliczamy Kinematykę Odwrotną dla zadanego punktu
            joint_angles = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, pos)
            
            # Zadajemy kąty na silniki robota
            for i in range(len(joint_angles)):
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_angles[i],
                                        force=100)
            
            # Pętla symulacyjna TYLKO dla ruchu między punktami.
            # Dajemy robotowi 120 kroków symulacji (ok. 0.5 sekundy) na dojechanie do celu.
            for _ in range(120):
                p.stepSimulation()
                time.sleep(1./240.)
                
        print("⏹ Sekwencja zakończona.")
        self.is_playing = False