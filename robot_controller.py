import json
import time
from pathlib import Path

import numpy as np
import pybullet as p


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
    JSON_FORMAT = "teach_and_play_motion"
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
    # Import / eksport JSON
    # ------------------------------------------------------------------

    def export_sequence(self, file_path: str | Path):
        """Zapisuje aktualnie nagraną trajektorię do pliku JSON."""
        if not self.waypoints:
            print("⚠ Brak zapisanych klatek do eksportu!")
            return False

        path = Path(file_path)
        payload = {
            "format": self.JSON_FORMAT,
            "version": self.JSON_VERSION,
            "frame_count": len(self.waypoints),
            "waypoints": [self._normalize_waypoint(frame) for frame in self.waypoints],
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"💾 Wyeksportowano {len(self.waypoints)} klatek do {path}.")
        return True

    def import_sequence(self, file_path: str | Path):
        """Wczytuje trajektorię z pliku JSON i zastępuje bieżącą sekwencję."""
        path = Path(file_path)
        if not path.exists():
            print(f"⚠ Nie znaleziono pliku importu: {path}")
            return False

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            imported_waypoints = self._parse_json_payload(payload)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"⚠ Nie udało się zaimportować sekwencji z {path}: {exc}")
            return False

        self.waypoints = imported_waypoints
        self.is_recording = False
        print(f"📂 Zaimportowano {len(self.waypoints)} klatek z {path}.")
        return True

    def _parse_json_payload(self, payload):
        """Waliduje zawartość pliku JSON i zwraca listę waypointów."""
        if not isinstance(payload, dict):
            raise ValueError("plik musi zawierać obiekt JSON")

        if payload.get("format") != self.JSON_FORMAT:
            raise ValueError(f"nieobsługiwany format pliku: {payload.get('format')!r}")

        if payload.get("version") != self.JSON_VERSION:
            raise ValueError(f"nieobsługiwana wersja pliku: {payload.get('version')!r}")

        raw_waypoints = payload.get("waypoints")
        if not isinstance(raw_waypoints, list):
            raise ValueError("pole 'waypoints' musi być listą")

        frame_count = payload.get("frame_count")
        if frame_count is not None and frame_count != len(raw_waypoints):
            raise ValueError("pole 'frame_count' nie zgadza się z liczbą klatek")

        normalized_waypoints = [self._normalize_waypoint(frame) for frame in raw_waypoints]
        self._validate_joint_counts(normalized_waypoints)
        return normalized_waypoints

    def _normalize_waypoint(self, frame):
        """Zwraca kanoniczną postać pojedynczej klatki ruchu."""
        if not isinstance(frame, dict):
            raise ValueError("każda klatka musi być obiektem JSON")

        angles = self._float_list(frame.get("angles"), "angles")
        cube_pos = self._float_list(frame.get("cube_pos"), "cube_pos", expected_len=3)
        gripper = frame.get("gripper")
        if not isinstance(gripper, bool):
            raise ValueError("pole 'gripper' musi być wartością bool")

        return {
            "angles": angles,
            "gripper": gripper,
            "cube_pos": cube_pos,
        }

    def _float_list(self, values, field_name: str, expected_len: int | None = None):
        if not isinstance(values, list):
            raise ValueError(f"pole '{field_name}' musi być listą")

        if expected_len is not None and len(values) != expected_len:
            raise ValueError(f"pole '{field_name}' musi mieć długość {expected_len}")

        normalized = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"pole '{field_name}' może zawierać tylko liczby")
            normalized.append(float(value))
        return normalized

    def _validate_joint_counts(self, waypoints: list[dict]):
        if not waypoints:
            return

        expected_count = len(waypoints[0]["angles"])
        if expected_count == 0:
            raise ValueError("pole 'angles' nie może być puste")

        for frame in waypoints:
            if len(frame["angles"]) != expected_count:
                raise ValueError("wszystkie klatki muszą mieć tę samą liczbę kątów")

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

        start_cube_pos = self.waypoints[0].get("cube_pos")
        if start_cube_pos is not None:
            p.resetBasePositionAndOrientation(cube_id, start_cube_pos, [0, 0, 0, 1])
            p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0])

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
