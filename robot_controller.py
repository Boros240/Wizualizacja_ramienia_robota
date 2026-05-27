import pybullet as p
import time
import numpy as np
import pygame


class TeachAndPlayController:
    """
    Kontroler trybu „Ucz i Odtwarzaj" (Teach & Play).
    """

    GRAB_THRESHOLD = 0.15  # dystans [m] do automatycznego chwytania przy odtwarzaniu

    def __init__(self, robot_id: int, end_effector_index: int):
        self.robot_id = robot_id
        self.ee_idx   = end_effector_index

        self.waypoints:  list[dict] = []
        self.is_playing  = False
        self.is_recording = False

    # ------------------------------------------------------------------
    # Nagrywanie
    # ------------------------------------------------------------------

    def toggle_recording(self):
        """Przełącza tryb ciągłego nagrywania (START / STOP)."""
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.waypoints.clear()
            print("\n NAGRYWANIE ROZPOCZeTE — ruszaj robotem!")
        else:
            print(f"\n NAGRYWANIE ZAKONCZONE — zapisano {len(self.waypoints)} klatek.")

    def record_frame(self, joint_angles: list, gripper_active: bool, cube_id: int):
        if not self.is_recording:
            return

        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        self.waypoints.append({
            "angles":  list(joint_angles),
            "gripper": gripper_active,
            "cube_pos": list(cube_pos),
        })

    def clear_sequence(self):
        """Czyści wszystkie zapisane waypoints i wyłącza nagrywanie."""
        self.waypoints.clear()
        self.is_recording = False
        print(" Pamięć sekwencji wyczyszczona.")

    # ------------------------------------------------------------------
    # Odtwarzanie
    # ------------------------------------------------------------------

    def play_sequence(self, cube_id: int):
        if not self.waypoints:
            print(" Brak zapisanych klatek do odtworzenia!")
            return

        print(f"\n Odtwarzam {len(self.waypoints)} klatek...")
        self.is_playing  = True
        constraint_id = None

        motor_channel = None
        if pygame.mixer.get_init():
            try:
                motor_sound = pygame.mixer.Sound("motor.wav")
                motor_channel = pygame.mixer.Channel(1)
                motor_channel.play(motor_sound, loops=-1)
            except:
                pass

        for step in self.waypoints:
            target_angles  = step["angles"]
            gripper_active = step["gripper"]

            for i, angle in enumerate(target_angles):
                p.setJointMotorControl2(
                    self.robot_id, i,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=200
                )

            if gripper_active and constraint_id is None:
                constraint_id = self._try_grab(cube_id)
            elif not gripper_active and constraint_id is not None:
                self._release(cube_id, constraint_id)
                constraint_id = None

            for _ in range(10):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

        if motor_channel:
            motor_channel.stop()

        print("Odtwarzanie zakonczone.")
        self.is_playing = False

    # ------------------------------------------------------------------
    # Prywatne — chwytak
    # ------------------------------------------------------------------

    def _is_close_to_cube(self, cube_id: int) -> bool:
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        return np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) < self.GRAB_THRESHOLD

    def _try_grab(self, cube_id: int) -> int | None:
        if not self._is_close_to_cube(cube_id):
            return None

        for i in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 0)

        ee_pos,   ee_orn   = p.getLinkState(self.robot_id, self.ee_idx)[0:2]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        return p.createConstraint(
            self.robot_id, self.ee_idx, cube_id, -1,
            p.JOINT_FIXED, [0, 0, 0], local_pos, [0, 0, 0], local_orn
        )

    def _release(self, cube_id: int, constraint_id: int):
        p.removeConstraint(constraint_id)
        for i in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 1)
