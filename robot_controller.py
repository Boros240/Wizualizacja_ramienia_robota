import pybullet as p
import time
import numpy as np

class TeachAndPlayController:
    def __init__(self, robot_id: int, end_effector_index: int):
        self.robot_id = robot_id
        self.ee_idx = end_effector_index
        self.waypoints = []
        self.is_playing = False
        self.is_recording = False # NOWA FLAGA

    def toggle_recording(self):
        """Wlacza i wylacza tryb ciaglego nagrywania."""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.waypoints.clear() # Czyścimy pamięć przy nowym nagraniu
            print("\n🔴 NAGRYWANIE ROZPOCZeTE... (Ruszaj robotem!)")
        else:
            print(f"\n⏹ NAGRYWANIE ZAKOnCZONE. Zapisano {len(self.waypoints)} klatek ruchu.")

    def record_frame(self, joint_angles: list, gripper_active: bool, cube_id: int):
        """Zapisuje pojedynczą klatkę ruchu (wywoływane automatycznie w tle)."""
        if not self.is_recording:
            return
            
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        self.waypoints.append({
            "angles": list(joint_angles),
            "gripper": gripper_active,
            "cube_pos": list(cube_pos)
        })

    def clear_sequence(self):
        self.waypoints.clear()
        self.is_recording = False
        print("Pamięć sekwencji wyczyszczona.")

    def _is_close_to_cube(self, cube_id: int, threshold: float = 0.15) -> bool:
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        dist = np.linalg.norm(np.array(cube_pos) - np.array(ee_pos))
        return dist < threshold

    def play_sequence(self, cube_id: int):
        """Odtwarza gęstą trajektorię klatka po klatce."""
        if not self.waypoints:
            print("Brak zapisanych punktów do odtworzenia!")
            return

        print(f"\n▶ Rozpoczynam odtwarzanie ({len(self.waypoints)} klatek)...")
        self.is_playing = True
        constraint_id = None

        for index, step in enumerate(self.waypoints):
            target_angles = step["angles"]
            gripper_active = step["gripper"]
            
            # Wymuszamy kąty bezpośrednio (bez linspace, bo klatki są gęsto upakowane)
            for i, angle in enumerate(target_angles):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=angle, force=200)

            # Logika Chwytaka
            if gripper_active and constraint_id is None and self._is_close_to_cube(cube_id):
                for i in range(-1, p.getNumJoints(self.robot_id)):
                    p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 0)
                
                ee_pos, ee_orn = p.getLinkState(self.robot_id, self.ee_idx)[0:2]
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
                local_cube_pos, local_cube_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)
                
                constraint_id = p.createConstraint(self.robot_id, self.ee_idx, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], local_cube_pos, [0, 0, 0], local_cube_orn)
                
            elif not gripper_active and constraint_id is not None:
                p.removeConstraint(constraint_id)
                constraint_id = None
                for i in range(-1, p.getNumJoints(self.robot_id)):
                    p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 1)
                
            # Skoro nagrywaliśmy co 10 kroków symulacji, odtwarzamy to z takim samym tempem
            for _ in range(10):
                p.stepSimulation()
                time.sleep(1./240.)
                
        print("⏹ Odtwarzanie zakończone.")
        self.is_playing = False