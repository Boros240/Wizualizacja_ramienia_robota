import time
from pathlib import Path

import pybullet as p
import pybullet_data
import numpy as np

from robot_controller import TeachAndPlayController

# ---------------------------------------------------------------------------
# Limity stawów w stopniach — spójne z wartościami w pliku simple_arm.urdf.
# Zmieniaj tutaj zamiast w URDF, jeśli chcesz ograniczyć zakres programowo.
#   Joint 0 (baza, oś Z):      ±150°  →  np. 30°–330° na skali 0–360°
#   Joint 1–3 (ramię, oś Y):   ±90°
# ---------------------------------------------------------------------------
JOINT_LIMITS_DEG = [
    (-150.0, 150.0),   # joint1 – obrót bazy
    ( -90.0,  90.0),   # joint2 – pierwsze ramię
    ( -90.0,  90.0),   # joint3 – drugie ramię
    ( -90.0,  90.0),   # joint_wrist – przegub
]
JOINT_LIMITS_RAD = [(np.deg2rad(lo), np.deg2rad(hi)) for lo, hi in JOINT_LIMITS_DEG]
MOTION_JSON_FILE = Path(__file__).with_name("robot_motion.json")


class RobotSimulation:
    EE_INDEX      = 4   # indeks ogniwa end-effektora w URDF
    NUM_JOINTS    = 4   # liczba sterowanych stawów
    GRAB_THRESHOLD = 0.15  # maksymalny dystans [m] do chwytania kostki
    USER_POINT_LIMITS = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
        "z": (0.05, 1.0),
    }

    def __init__(self):
        self._setup_physics()
        self._load_models()
        self._setup_ui()

        self.controller = TeachAndPlayController(self.robot, self.EE_INDEX)

        self.current_angles   = [0.0] * self.NUM_JOINTS
        self.prev_slider_vals = [0.3, 0.0, 0.4]
        self.gripper_active   = False
        self.constraint_id    = None
        self.space_pressed    = False
        self.tick_counter     = 0  # do regulacji częstotliwości nagrywania

    # ------------------------------------------------------------------
    # Inicjalizacja
    # ------------------------------------------------------------------

    def _setup_physics(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def _load_models(self):
        self.plane = p.loadURDF("plane.urdf")
        self.cube  = p.loadURDF("cube.urdf",       [0.3, 0.0, 0.2], useFixedBase=False)
        self.robot = p.loadURDF("simple_arm.urdf", [0,   0,   0  ], useFixedBase=True)

    def _setup_ui(self):
        self.sliders = {
            'x': p.addUserDebugParameter("Ramię Cel X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Ramię Cel Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Ramię Cel Z",  0.1, 1.0, 0.4),
        }
        self.cube_sliders = {
            'x': p.addUserDebugParameter("Start Kostki X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Start Kostki Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Start Kostki Z",  0.05, 1.0, 0.2),
        }
        self.buttons = {
            'set_cube': p.addUserDebugParameter("USTAW KOSTKĘ NA STARCIE",  1, 0, 0),
            'transfer_cube': p.addUserDebugParameter(" PRZENIEŚ KOSTKĘ A -> B", 1, 0, 0),
            'record':   p.addUserDebugParameter(" NAGRYWAJ (START/STOP)", 1, 0, 0),
            'play':     p.addUserDebugParameter(" ODTWÓRZ SEKWENCJĘ",     1, 0, 0),
            'import':   p.addUserDebugParameter(" IMPORTUJ JSON",          1, 0, 0),
            'export':   p.addUserDebugParameter(" EKSPORTUJ JSON",         1, 0, 0),
            'clear':    p.addUserDebugParameter(" WYCZYŚĆ PAMIĘĆ",        1, 0, 0),
        }
        self.btn_states = {k: 0 for k in self.buttons}

    # ------------------------------------------------------------------
    # Obsługa wejść użytkownika
    # ------------------------------------------------------------------

    def _check_button(self, btn_name: str) -> bool:
        """Zwraca True dokładnie raz po każdym kliknięciu przycisku."""
        current_clicks = p.readUserDebugParameter(self.buttons[btn_name])
        if current_clicks > self.btn_states[btn_name]:
            self.btn_states[btn_name] = current_clicks
            return True
        return False

    def _clamp_angles(self):
        """
        Przycina current_angles do limitów zdefiniowanych w JOINT_LIMITS_RAD.

        BEZ tego kąty mogłyby akumulować się poza fizycznymi limitami URDF:
        robot stoi w miejscu, ale zmienna rośnie → żeby robot ruszył w drugą
        stronę, trzeba najpierw „odwinąć" tę nadwyżkę. Efekt: pozorna blokada.
        """
        for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            self.current_angles[i] = float(np.clip(self.current_angles[i], lo, hi))

    def handle_ik_sliders(self):
        """Sterowanie przez suwaki IK — aktualizuje kąty tylko przy zmianie."""
        vals = [p.readUserDebugParameter(self.sliders[k]) for k in ('x', 'y', 'z')]
        if any(abs(v - pv) > 0.001 for v, pv in zip(vals, self.prev_slider_vals)):
            ik_angles = p.calculateInverseKinematics(self.robot, self.EE_INDEX, vals)
            self.current_angles   = list(ik_angles)[:self.NUM_JOINTS]
            self.prev_slider_vals = vals
            self._clamp_angles()  # IK może zwrócić kąt spoza limitu

    def handle_keyboard(self):
        """
        Sterowanie klawiaturą:
            : obrót bazy    (joint 0)
            : pierwsze ramię (joint 1)
          Z / X : drugie ramię  (joint 2)
          C / V : przegub       (joint 3)
          SPACJA: chwytak (toggle)
        """
        keys  = p.getKeyboardEvents()
        delta = 0.01  # krok kąta na klatkę [rad]

        key_map = {
            p.B3G_LEFT_ARROW:  (0, -delta),
            p.B3G_RIGHT_ARROW: (0, +delta),
            p.B3G_UP_ARROW:    (1, -delta),
            p.B3G_DOWN_ARROW:  (1, +delta),
            ord('z'):          (2, +delta),
            ord('x'):          (2, -delta),
            ord('c'):          (3, +delta),
            ord('v'):          (3, -delta),
        }

        for key, (joint_idx, d) in key_map.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                self.current_angles[joint_idx] += d

        # Przycinamy po wszystkich naciśnięciach — to naprawia "blokadę" ruchu
        self._clamp_angles()

        # Toggle chwytaka na pierwsze wciśnięcie spacji
        space_down = ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN
        if space_down and not self.space_pressed:
            self._toggle_gripper()
        self.space_pressed = space_down

    # ------------------------------------------------------------------
    # Logika chwytaka
    # ------------------------------------------------------------------

    def _toggle_gripper(self):
        self.gripper_active = not self.gripper_active

        if self.gripper_active and self.constraint_id is None:
            self._try_grab()
        elif not self.gripper_active and self.constraint_id is not None:
            self._release()

    def _try_grab(self):
        """Próbuje przypiąć kostkę do end-effektora jeśli jest wystarczająco blisko."""
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        ee_pos   = p.getLinkState(self.robot, self.EE_INDEX)[0]

        if np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) >= self.GRAB_THRESHOLD:
            self.gripper_active = False  # za daleko — anuluj
            return

        # Wyłącz kolizje robot–kostka podczas trzymania
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 0)

        # Oblicz pozycję kostki w układzie end-effektora
        ee_pos,   ee_orn   = p.getLinkState(self.robot, self.EE_INDEX)[0:2]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        self.constraint_id = p.createConstraint(
            self.robot, self.EE_INDEX, self.cube, -1,
            p.JOINT_FIXED, [0, 0, 0], local_pos, [0, 0, 0], local_orn
        )

    def _release(self):
        """Odpina kostkę i przywraca kolizje."""
        p.removeConstraint(self.constraint_id)
        self.constraint_id = None
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    # ------------------------------------------------------------------
    # Narzędzia
    # ------------------------------------------------------------------

    def _clamp_cartesian_point(self, point):
        """Przycina punkt XYZ do bezpiecznego obszaru roboczego robota."""
        clamped_point = []
        for idx, axis in enumerate(("x", "y", "z")):
            lo, hi = self.USER_POINT_LIMITS[axis]
            clamped_point.append(float(np.clip(point[idx], lo, hi)))
        return clamped_point

    def _read_point_from_user(self, point_name: str):
        """Pobiera od użytkownika punkt XYZ w formacie 'x y z' lub 'x,y,z'."""
        while True:
            raw_value = input(
                f"Podaj współrzędne punktu {point_name} [x y z] "
                f"w zakresie x/y ({self.USER_POINT_LIMITS['x'][0]}..{self.USER_POINT_LIMITS['x'][1]}), "
                f"z ({self.USER_POINT_LIMITS['z'][0]}..{self.USER_POINT_LIMITS['z'][1]}): "
            ).strip()
            normalized = raw_value.replace(",", " ")
            parts = [part for part in normalized.split(" ") if part]

            if len(parts) != 3:
                print("⚠ Niepoprawny format. Wpisz dokładnie trzy liczby, np. 0.2 0.1 0.2")
                continue

            try:
                values = [float(part) for part in parts]
            except ValueError:
                print("⚠ Dozwolone są wyłącznie liczby, np. 0.2 0.1 0.2")
                continue

            clamped = self._clamp_cartesian_point(values)
            if any(abs(a - b) > 1e-9 for a, b in zip(values, clamped)):
                print(f"ℹ Punkt {point_name} został przycięty do obszaru roboczego: {clamped}")

            return clamped

    def _move_end_effector_to(self, target_xyz, duration_s: float = 1.0):
        """Przemieszcza end-effektor do punktu XYZ płynnie w przestrzeni stawów."""
        target_xyz = self._clamp_cartesian_point(target_xyz)
        ik_solution = p.calculateInverseKinematics(self.robot, self.EE_INDEX, target_xyz)
        target_angles = list(ik_solution)[:self.NUM_JOINTS]
        for idx, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            target_angles[idx] = float(np.clip(target_angles[idx], lo, hi))

        start_angles = list(self.current_angles)
        steps = max(1, int(duration_s * 240))
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated = [
                start + (target - start) * alpha
                for start, target in zip(start_angles, target_angles)
            ]
            for joint_idx, angle in enumerate(interpolated):
                p.setJointMotorControl2(
                    self.robot,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=200,
                    maxVelocity=1.5,
                )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        self.current_angles = target_angles
        self._clamp_angles()

    def _set_gripper_state(self, active: bool):
        """
        Ustawia stan chwytaka:
          active=True  -> próbuje chwycić kostkę i zwraca True/False
          active=False -> zwalnia kostkę
        """
        if active:
            if self.constraint_id is None:
                self.gripper_active = True
                self._try_grab()
            grab_succeeded = self.constraint_id is not None
            self.gripper_active = grab_succeeded
            return grab_succeeded

        if self.constraint_id is not None:
            self._release()
        self.gripper_active = False
        return True

    def transfer_cube_from_a_to_b(self, point_a, point_b):
        """
        Sekwencja: podejście nad A -> zejście do A -> chwyt -> podniesienie
        -> ruch nad B -> zejście do B -> odłożenie -> odejście od kostki.
        """
        point_a = self._clamp_cartesian_point(point_a)
        point_b = self._clamp_cartesian_point(point_b)

        hover_height = 0.18
        point_a_hover = self._clamp_cartesian_point([point_a[0], point_a[1], point_a[2] + hover_height])
        point_b_hover = self._clamp_cartesian_point([point_b[0], point_b[1], point_b[2] + hover_height])

        # Ustaw kostkę dokładnie w punkcie A, aby ruch był deterministyczny.
        p.resetBasePositionAndOrientation(self.cube, point_a, [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])
        self._set_gripper_state(False)

        print(f"▶ Transfer kostki: A={point_a} -> B={point_b}")
        self._move_end_effector_to(point_a_hover, duration_s=0.9)
        self._move_end_effector_to(point_a, duration_s=0.7)

        if not self._set_gripper_state(True):
            print("⚠ Nie udało się chwycić kostki w punkcie A.")
            self._move_end_effector_to(point_a_hover, duration_s=0.6)
            return False

        self._move_end_effector_to(point_a_hover, duration_s=0.8)
        self._move_end_effector_to(point_b_hover, duration_s=1.1)
        self._move_end_effector_to(point_b, duration_s=0.7)
        self._set_gripper_state(False)
        self._move_end_effector_to(point_b_hover, duration_s=0.7)
        print("✅ Transfer kostki zakończony.")
        return True

    def transfer_cube_from_user_points(self):
        """Pobiera punkty A/B od użytkownika i uruchamia transfer kostki."""
        print("\n=== TRYB A -> B ===")
        point_a = self._read_point_from_user("A")
        point_b = self._read_point_from_user("B")
        self.transfer_cube_from_a_to_b(point_a, point_b)

    def reset_cube_position(self):
        """Resetuje pozycję kostki do wartości z suwaków."""
        cx, cy, cz = [p.readUserDebugParameter(self.cube_sliders[k]) for k in ('x', 'y', 'z')]
        p.resetBasePositionAndOrientation(self.cube, [cx, cy, cz], [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

    def sync_state_after_play(self):
        """Po odtworzeniu synchronizuje stan kontrolera z ostatnim waypointem."""
        if not self.controller.waypoints:
            return
        last = self.controller.waypoints[-1]
        self.current_angles = list(last["angles"])
        self.gripper_active  = last["gripper"]
        if not self.gripper_active and self.constraint_id is not None:
            self._release()

    # ------------------------------------------------------------------
    # Główna pętla
    # ------------------------------------------------------------------

    def run(self):
        print("====== SYMULACJA ROBOTA — TRYB NAGRYWANIA ======")
        print("Klawiatura:  baza |  ramię1 | Z/X ramię2 | C/V przegub | SPACJA chwytak")
        print(f"Import/eksport JSON używa pliku: {MOTION_JSON_FILE}")

        while True:
            self.tick_counter += 1

            # 1. Odczyt wejść
            self.handle_ik_sliders()
            self.handle_keyboard()

            # 2. Przyciski UI
            if self._check_button('set_cube'):
                self.reset_cube_position()

            if self._check_button('transfer_cube'):
                self.transfer_cube_from_user_points()

            if self._check_button('record'):
                self.controller.toggle_recording()

            if self._check_button('play'):
                self.reset_cube_position()
                self.controller.play_sequence(self.cube)
                self.sync_state_after_play()

            if self._check_button('import'):
                self.controller.import_sequence(MOTION_JSON_FILE)

            if self._check_button('export'):
                self.controller.export_sequence(MOTION_JSON_FILE)

            if self._check_button('clear'):
                self.controller.clear_sequence()

            # 3. Ciągłe nagrywanie w tle (co 10 kroków ≈ 24 klatki/s)
            if self.controller.is_recording and self.tick_counter % 10 == 0:
                self.controller.record_frame(self.current_angles, self.gripper_active, self.cube)

            # 4. Fizyka — steruj silnikami tylko gdy nie odtwarzamy
            if not self.controller.is_playing:
                for i, angle in enumerate(self.current_angles):
                    p.setJointMotorControl2(
                        self.robot, i,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=200,
                        maxVelocity=1.5
                    )

            p.stepSimulation()
            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    app = RobotSimulation()
    app.run()
