import pybullet as p
import time
import numpy as np # Importujemy numpy do ułatwienia obliczeń matematycznych

class TeachAndPlayController:
    def __init__(self, robot_id, end_effector_index):
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.waypoints = []
        self.is_playing = False

    def save_waypoint(self, joint_angles, gripper_active, cube_id):
        """Zapisuje kąty stawów, stan chwytaka i POZYCJĘ KOSTKI."""
        # Pobieramy pozycję kostki
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
        
        step_data = {
            "angles": list(joint_angles),
            "gripper": gripper_active,
            "cube_pos": list(cube_pos) # Zapisujemy pozycję X, Y, Z kostki
        }
        self.waypoints.append(step_data)
        
        stan = "ZŁAPANY" if gripper_active else "PUSZCZONY"
        print(f"[{len(self.waypoints)}] Zapisano krok. Chwytak: {stan} | Pozycja kostki: {cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f}")

    def clear_sequence(self):
        self.waypoints.clear()
        print("Pamięć sekwencji została wyczyszczona.")

    def play_sequence(self, cube_id):
        if not self.waypoints:
            print("Brak zapisanych punktów do odtworzenia!")
            return

        print("\n▶ Rozpoczynam odtwarzanie sekwencji...")
        self.is_playing = True
        constraint_id = None
        
        # Pobieramy aktualne kąty złączy przed rozpoczęciem ruchu
        num_joints = p.getNumJoints(self.robot_id)
        current_angles = []
        # Uwaga: zakładamy, że indeksy złączy którymi sterujemy zaczynają się od 0
        for i in range(len(self.waypoints[0]["angles"])):
            joint_state = p.getJointState(self.robot_id, i)
            current_angles.append(joint_state[0])

        for index, step in enumerate(self.waypoints):
            print(f"Odtwarzam krok {index + 1}/{len(self.waypoints)}")
            target_angles = step["angles"]
            gripper_active = step["gripper"]
            
            # --- 1. PŁYNNY RUCH (Interpolacja liniowa kątów) ---
            num_steps = 100 # W ilu krokach chcemy dotrzeć do celu (wpływa na płynność)
            
            # Generujemy ścieżkę (trajektorię) od aktualnych kątów do docelowych
            trajectory = np.linspace(current_angles, target_angles, num_steps)
            
            for path_point in trajectory:
                for i in range(len(path_point)):
                    p.setJointMotorControl2(bodyIndex=self.robot_id,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=path_point[i],
                                            force=200,
                                            maxVelocity=1.5) # Ograniczamy prędkość dla lepszego efektu
                
                # Krok symulacji podczas ruchu do punktu
                p.stepSimulation()
                time.sleep(1./240.)
                
            # Zapisujemy, że osiągnęliśmy cel, aby następny ruch zaczął się stąd
            current_angles = target_angles

            # --- 2. OBSŁUGA CHWYTAKA PO DOJAZDZIE DO PUNKTU ---
            if gripper_active and constraint_id is None:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
                ee_pos = ee_state[0]
                dist = sum([(a - b)**2 for a, b in zip(cube_pos, ee_pos)])**0.5
                
                if dist < 0.15:
                    for i in range(-1, p.getNumJoints(self.robot_id)):
                        p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, enableCollision=0)
                    
                    # NOWE: Matematyka transformacji dla odtwarzania
                    ee_pos, ee_orn = ee_state[0], ee_state[1]
                    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
                    
                    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
                    local_cube_pos, local_cube_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)
                    
                    constraint_id = p.createConstraint(parentBodyUniqueId=self.robot_id,
                                                       parentLinkIndex=self.end_effector_index,
                                                       childBodyUniqueId=cube_id,
                                                       childLinkIndex=-1,
                                                       jointType=p.JOINT_FIXED,
                                                       jointAxis=[0, 0, 0],
                                                       parentFramePosition=local_cube_pos,
                                                       childFramePosition=[0, 0, 0],
                                                       parentFrameOrientation=local_cube_orn)
                    print("   [Kostka złapana poprawnie]")
                    
            elif not gripper_active and constraint_id is not None:
                p.removeConstraint(constraint_id)
                constraint_id = None
                
                # PRZYWRACAMY KOLIZJE Z CAŁYM ROBOTEM
                for i in range(-1, p.getNumJoints(self.robot_id)):
                    p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, enableCollision=1)
                print("   [Kostka puszczona]")

            # Mała pauza po dojechaniu i (ewentualnej) akcji chwytaka
            for _ in range(60): # 0.25 sekundy przerwy
                p.stepSimulation()
                time.sleep(1./240.)
                
        print("⏹ Sekwencja zakończona.")
        self.is_playing = False