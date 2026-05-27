import pybullet as p
import time
import numpy as np


class TeachAndPlayController:
    """
    Kontroler trybu „Ucz i Odtwarzaj" (Teach & Play).

    Użycie:
      1. toggle_recording() — włącz nagrywanie
      2. record_frame(...)  — wywołuj co klatkę z logic.py
      3. toggle_recording() — zatrzymaj nagrywanie
      4. play_sequence(...)  — odtwórz zapisaną trajektorię
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
            print("\n🔴 NAGRYWANIE ROZPOCZĘTE — ruszaj robotem!")
        else:
            print(f"\n⏹ NAGRYWANIE ZAKOŃCZONE — zapisano {len(self.waypoints)} klatek.")

    def record_frame(self, joint_angles: list, gripper_active: bool, cube_id: int):
        """
        Zapisuje pojedynczą klatkę ruchu.
        Wywoływana automatycznie z pętli w logic.py — nie wywołuj ręcznie.
        """
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
        print("🗑 Pamięć sekwencji wyczyszczona.")

    # ------------------------------------------------------------------
    # Odtwarzanie
    # ------------------------------------------------------------------

    def play_sequence(self, cube_id: int):
        """
        Odtwarza zapisaną trajektorię klatka po klatce.
        Klatki są gęsto upakowane (≈24/s), więc nie ma potrzeby interpolacji.
        """
        if not self.waypoints:
            print("⚠ Brak zapisanych klatek do odtworzenia!")
            return

        print(f"\n▶ Odtwarzam {len(self.waypoints)} klatek...")
        self.is_playing  = True
        constraint_id = None

        for step in self.waypoints:
            target_angles  = step["angles"]
            gripper_active = step["gripper"]

            # Ustaw kąty stawów bezpośrednio
            for i, angle in enumerate(target_angles):
                p.setJointMotorControl2(
                    self.robot_id, i,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=200
                )

            # Logika chwytaka
            if gripper_active and constraint_id is None:
                constraint_id = self._try_grab(cube_id)
            elif not gripper_active and constraint_id is not None:
                self._release(cube_id, constraint_id)
                constraint_id = None

            # Odtwarzaj z tym samym tempem co nagranie (co 10 kroków symulacji)
            for _ in range(10):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

        print("⏹ Odtwarzanie zakończone.")
        self.is_playing = False

    # ------------------------------------------------------------------
    # Prywatne — chwytak
    # ------------------------------------------------------------------

    def _is_close_to_cube(self, cube_id: int) -> bool:
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        return np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) < self.GRAB_THRESHOLD

    def _try_grab(self, cube_id: int) -> int | None:
        """
        Tworzy więź (constraint) między end-effektorem a kostką.
        Zwraca ID więzi lub None jeśli kostka jest za daleko.
        """
        if not self._is_close_to_cube(cube_id):
            return None

        # Wyłącz kolizje robot–kostka podczas trzymania
        for i in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 0)

        # Oblicz lokalną pozycję kostki względem end-effektora
        ee_pos,   ee_orn   = p.getLinkState(self.robot_id, self.ee_idx)[0:2]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        return p.createConstraint(
            self.robot_id, self.ee_idx, cube_id, -1,
            p.JOINT_FIXED, [0, 0, 0], local_pos, [0, 0, 0], local_orn
        )

    def _release(self, cube_id: int, constraint_id: int):
        """Usuwa więź i przywraca kolizje robot–kostka."""
        p.removeConstraint(constraint_id)
        for i in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, cube_id, i, -1, 1)
