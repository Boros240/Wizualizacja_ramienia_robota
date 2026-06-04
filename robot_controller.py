import pybullet as p
import time
import json
import os
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
    # Import / eksport sekwencji do pliku JSON
    # ------------------------------------------------------------------

    def save_to_json(self, filepath: str = "sequence.json") -> bool:
        """Zapisuje nagraną sekwencję ruchów do pliku JSON.

        Format pliku jest czytelny dla człowieka i przenośny — pozwala
        przenieść „nauczony" program między sesjami symulacji.
        """
        if not self.waypoints:
            print(" Brak klatek do zapisania — najpierw coś nagraj.")
            return False

        payload = {
            "version": 1,
            "num_joints": len(self.waypoints[0]["angles"]),
            "frame_count": len(self.waypoints),
            "waypoints": self.waypoints,
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f" Błąd zapisu do '{filepath}': {exc}")
            return False

        print(f" Zapisano {len(self.waypoints)} klatek do '{filepath}'.")
        return True

    def load_from_json(self, filepath: str = "sequence.json") -> bool:
        """Wczytuje sekwencję ruchów z pliku JSON, zastępując bieżącą."""
        if not os.path.exists(filepath):
            print(f" Plik '{filepath}' nie istnieje.")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f" Błąd odczytu '{filepath}': {exc}")
            return False

        waypoints = payload.get("waypoints", [])
        if not self._validate_waypoints(waypoints):
            print(f" Plik '{filepath}' ma nieprawidłowy format sekwencji.")
            return False

        self.is_recording = False
        self.waypoints = [
            {
                "angles": [float(a) for a in wp["angles"]],
                "gripper": bool(wp["gripper"]),
                "cube_pos": [float(c) for c in wp.get("cube_pos", [0.0, 0.0, 0.0])],
            }
            for wp in waypoints
        ]
        print(f" Wczytano {len(self.waypoints)} klatek z '{filepath}'.")
        return True

    @staticmethod
    def _validate_waypoints(waypoints) -> bool:
        """Sprawdza, czy struktura wczytanych klatek jest poprawna."""
        if not isinstance(waypoints, list) or not waypoints:
            return False
        for wp in waypoints:
            if not isinstance(wp, dict):
                return False
            if "angles" not in wp or "gripper" not in wp:
                return False
            if not isinstance(wp["angles"], list) or not wp["angles"]:
                return False
        return True

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


class PickAndPlaceController:
    """Realizuje automatyczną sekwencję „pobierz i odłóż" między dwoma
    punktami A i B.

    Koncepcja: w punkcie A ramię chwyta kostkę, przenosi ją nad punkt B
    i tam opuszcza. Klasa odpowiada wyłącznie za *plan* ruchu — fizyczne
    przemieszczanie ramienia oraz chwytanie/puszczanie kostki realizują
    funkcje (callbacki) przekazane przy tworzeniu obiektu. Dzięki temu
    logika zadania jest odseparowana od silnika fizyki (zasada pojedynczej
    odpowiedzialności).
    """

    def __init__(self, move_fn, grab_fn, release_fn, approach_height: float = 0.25):
        self._move = move_fn        # move_fn(pozycja_xyz) -> przesuwa końcówkę
        self._grab = grab_fn        # grab_fn() -> próbuje chwycić kostkę
        self._release = release_fn  # release_fn() -> puszcza kostkę
        self.approach_height = approach_height

        self.point_a: list | None = None
        self.point_b: list | None = None
        self.is_running = False

    def set_points(self, point_a, point_b):
        """Ustawia punkt pobrania (A) i punkt odłożenia (B)."""
        self.point_a = list(point_a)
        self.point_b = list(point_b)

    def execute(self) -> bool:
        """Wykonuje pełną sekwencję A → B (pobranie i odłożenie)."""
        if self.point_a is None or self.point_b is None:
            print(" Najpierw ustaw punkty A i B.")
            return False
        if self.is_running:
            return False

        self.is_running = True
        a, b = self.point_a, self.point_b
        above_a = [a[0], a[1], a[2] + self.approach_height]
        above_b = [b[0], b[1], b[2] + self.approach_height]

        print(f"\n PICK & PLACE:  A={[round(v, 2) for v in a]}  ->  "
              f"B={[round(v, 2) for v in b]}")

        # 1. Podejście nad punkt A i opuszczenie do kostki
        self._move(above_a)
        self._move(a)
        # 2. Chwyt kostki
        self._grab()
        # 3. Podniesienie i transport nad punkt B
        self._move(above_a)
        self._move(above_b)
        # 4. Opuszczenie i puszczenie kostki
        self._move(b)
        self._release()
        # 5. Odsunięcie ramienia w górę
        self._move(above_b)

        print(" PICK & PLACE zakończone.")
        self.is_running = False
        return True
