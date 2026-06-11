import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pygame


class TeachAndPlayController:
    """
    Kontroler trybu „Ucz i Odtwarzaj" (Teach & Play).
    """

    GRAB_THRESHOLD = 0.15  # dystans [m] do automatycznego chwytania przy odtwarzaniu
    DEFAULT_SEQUENCE_FILE = "teach_play_sequence.json"
    JSON_VERSION = 1

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
    # Import / eksport
    # ------------------------------------------------------------------

    def export_sequence(self, file_path: str | Path = DEFAULT_SEQUENCE_FILE) -> Path:
        """Zapisuje aktualny program Teach & Play do pliku JSON."""
        path = Path(file_path)
        data = {
            "version": self.JSON_VERSION,
            "format": "teach_and_play_sequence",
            "waypoint_count": len(self.waypoints),
            "waypoints": self.waypoints,
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f" Wyeksportowano {len(self.waypoints)} klatek do pliku: {path}")
        return path

    def import_sequence(self, file_path: str | Path = DEFAULT_SEQUENCE_FILE) -> int:
        """Wczytuje program Teach & Play z pliku JSON."""
        path = Path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        raw_waypoints = data.get("waypoints") if isinstance(data, dict) else data
        if not isinstance(raw_waypoints, list):
            raise ValueError("Plik JSON nie zawiera listy 'waypoints'.")

        waypoints = [self._validate_waypoint(step, index) for index, step in enumerate(raw_waypoints, start=1)]
        self.waypoints = waypoints
        self.is_recording = False
        print(f" Zaimportowano {len(self.waypoints)} klatek z pliku: {path}")
        return len(self.waypoints)

    @staticmethod
    def _validate_waypoint(step: dict, index: int) -> dict:
        if not isinstance(step, dict):
            raise ValueError(f"Klatka #{index} nie jest obiektem JSON.")

        try:
            angles = [float(value) for value in step["angles"]]
            cube_pos = [float(value) for value in step.get("cube_pos", [0.3, 0.0, 0.2])]
            gripper = bool(step["gripper"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Klatka #{index} ma niepoprawny format.") from exc

        if not angles:
            raise ValueError(f"Klatka #{index} nie zawiera kątów stawów.")
        if len(cube_pos) != 3:
            raise ValueError(f"Klatka #{index} musi mieć cube_pos jako [x, y, z].")

        return {
            "angles": angles,
            "gripper": gripper,
            "cube_pos": cube_pos,
        }

    # ------------------------------------------------------------------
    # Program A -> B
    # ------------------------------------------------------------------

    def build_pick_and_place_sequence(
        self,
        point_a: list[float],
        point_b: list[float],
        num_joints: int,
        joint_limits: list[tuple[float, float]] | None = None,
        approach_height: float = 0.25,
        samples_per_segment: int = 20,
    ) -> int:
        """
        Generuje program: dojazd do A, chwyt, przeniesienie do B i puszczenie.
        Punkty A/B oznaczają pozycje środka kostki.
        """
        point_a = self._validate_point(point_a, "A")
        point_b = self._validate_point(point_b, "B")

        above_a = [point_a[0], point_a[1], point_a[2] + approach_height]
        above_b = [point_b[0], point_b[1], point_b[2] + approach_height]

        program: list[dict] = []
        self._append_motion_segment(program, above_a, point_a, False, point_a, num_joints, joint_limits, samples_per_segment)
        self._append_hold_frames(program, point_a, True, point_a, num_joints, joint_limits, frames=12)
        self._append_motion_segment(program, point_a, above_a, True, point_a, num_joints, joint_limits, samples_per_segment)
        self._append_motion_segment(program, above_a, above_b, True, point_a, num_joints, joint_limits, samples_per_segment)
        self._append_motion_segment(program, above_b, point_b, True, point_b, num_joints, joint_limits, samples_per_segment)
        self._append_hold_frames(program, point_b, False, point_b, num_joints, joint_limits, frames=12)
        self._append_motion_segment(program, point_b, above_b, False, point_b, num_joints, joint_limits, samples_per_segment)

        self.waypoints = program
        self.is_recording = False
        print(f" Wygenerowano program A -> B ({len(self.waypoints)} klatek).")
        return len(self.waypoints)

    @staticmethod
    def _validate_point(point: list[float], name: str) -> list[float]:
        if point is None:
            raise ValueError(f"Punkt {name} nie został ustawiony.")
        if len(point) != 3:
            raise ValueError(f"Punkt {name} musi mieć format [x, y, z].")
        return [float(value) for value in point]

    def _append_motion_segment(
        self,
        program: list[dict],
        start: list[float],
        end: list[float],
        gripper: bool,
        cube_pos: list[float],
        num_joints: int,
        joint_limits: list[tuple[float, float]] | None,
        samples: int,
    ):
        for step in range(samples):
            alpha = step / max(samples - 1, 1)
            pos = self._interpolate_point(start, end, alpha)
            program.append(self._make_waypoint(pos, gripper, cube_pos, num_joints, joint_limits))

    def _append_hold_frames(
        self,
        program: list[dict],
        pos: list[float],
        gripper: bool,
        cube_pos: list[float],
        num_joints: int,
        joint_limits: list[tuple[float, float]] | None,
        frames: int,
    ):
        waypoint = self._make_waypoint(pos, gripper, cube_pos, num_joints, joint_limits)
        for _ in range(frames):
            program.append(dict(waypoint))

    @staticmethod
    def _interpolate_point(start: list[float], end: list[float], alpha: float) -> list[float]:
        return [s + (e - s) * alpha for s, e in zip(start, end)]

    def _make_waypoint(
        self,
        pos: list[float],
        gripper: bool,
        cube_pos: list[float],
        num_joints: int,
        joint_limits: list[tuple[float, float]] | None,
    ) -> dict:
        angles = list(p.calculateInverseKinematics(self.robot_id, self.ee_idx, pos))[:num_joints]
        if joint_limits:
            angles = [float(np.clip(angle, low, high)) for angle, (low, high) in zip(angles, joint_limits)]

        return {
            "angles": angles,
            "gripper": gripper,
            "cube_pos": list(cube_pos),
        }

    # ------------------------------------------------------------------
    # Odtwarzanie
    # ------------------------------------------------------------------

    def play_sequence(self, cube_id: int) -> int | None:
        if not self.waypoints:
            print(" Brak zapisanych klatek do odtworzenia!")
            return None

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
        return constraint_id

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

    def __init__(self, move_fn, grab_fn, release_fn,
                 approach_height: float = 0.45, grab_clearance: float = 0.07):
        self._move = move_fn        # move_fn(pozycja_xyz) -> przesuwa końcówkę
        self._grab = grab_fn        # grab_fn() -> próbuje chwycić kostkę
        self._release = release_fn  # release_fn() -> puszcza kostkę
        # Wysokość, na jaką ramię podchodzi NAD punkt przed opuszczeniem.
        self.approach_height = approach_height
        # Wysokość zatrzymania końcówki nad punktem przy chwytaniu/odkładaniu
        # (ramię nie sięga do samej podłogi, więc „zawisa" tuż nad kostką).
        self.grab_clearance = grab_clearance

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
        down_a  = [a[0], a[1], a[2] + self.grab_clearance]
        down_b  = [b[0], b[1], b[2] + self.grab_clearance]

        print(f"\n PICK & PLACE:  A={[round(v, 2) for v in a]}  ->  "
              f"B={[round(v, 2) for v in b]}")

        # 1. Podejście z góry nad punkt A i opuszczenie do kostki
        self._move(above_a)
        self._move(down_a)
        # 2. Chwyt kostki
        self._grab()
        # 3. Podniesienie i transport nad punkt B
        self._move(above_a)
        self._move(above_b)
        # 4. Opuszczenie i puszczenie kostki
        self._move(down_b)
        self._release()
        # 5. Odsunięcie ramienia w górę
        self._move(above_b)

        print(" PICK & PLACE zakończone.")
        self.is_running = False
        return True
