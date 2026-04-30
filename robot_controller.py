import pybullet as p
import time

class TeachAndPlayController:
    def __init__(self, robot_id, end_effector_index):
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.waypoints = []  # Lista przechowująca słowniki: kąty stawów i stan chwytaka
        self.is_playing = False 

    def save_waypoint(self, joint_angles, gripper_active):
        """Zapisuje aktualne kąty stawów i stan chwytaka."""
        step_data = {
            "angles": list(joint_angles),
            "gripper": gripper_active
        }
        self.waypoints.append(step_data)
        stan = "ZŁAPANY" if gripper_active else "PUSZCZONY"
        print(f"[{len(self.waypoints)}] Zapisano krok. Chwytak: {stan}")

    def clear_sequence(self):
        """Czyści pamięć ruchów."""
        self.waypoints.clear()
        print("Pamięć sekwencji została wyczyszczona.")

    def play_sequence(self, cube_id):
        """Odtwarza zapisaną sekwencję krok po kroku z obsługą kostki."""
        if not self.waypoints:
            print("Brak zapisanych punktów do odtworzenia!")
            return

        print("▶ Rozpoczynam odtwarzanie sekwencji...")
        self.is_playing = True
        constraint_id = None
        
        for index, step in enumerate(self.waypoints):
            print(f"Odtwarzam krok {index + 1}/{len(self.waypoints)}")
            angles = step["angles"]
            gripper_active = step["gripper"]
            
            # Zadajemy kąty na silniki robota
            for i in range(len(angles)):
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=angles[i],
                                        force=200)
            
            # --- OBSŁUGA CHWYTAKA W TRAKCIE ODTWARZANIA ---
            if gripper_active and constraint_id is None:
                # Sprawdzamy odległość, żeby ramię nie złapało kostki z drugiego końca mapy na siłę
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
                ee_pos = ee_state[0]
                dist = sum([(a - b)**2 for a, b in zip(cube_pos, ee_pos)])**0.5
                
                if dist < 0.15: # Jeśli jesteśmy wystarczająco blisko kostki
                    constraint_id = p.createConstraint(self.robot_id, self.end_effector_index, 
                                                       cube_id, -1, p.JOINT_FIXED, 
                                                       [0, 0, 0], [0, 0, 0], [0, 0, 0])
            elif not gripper_active and constraint_id is not None:
                # Puszczamy kostkę
                p.removeConstraint(constraint_id)
                constraint_id = None
                
            # Dajemy robotowi chwilę na dojechanie do pozycji (np. 120 kroków symulacji = 0.5s)
            for _ in range(120):
                p.stepSimulation()
                time.sleep(1./240.)
                
        print("⏹ Sekwencja zakończona.")
        self.is_playing = False