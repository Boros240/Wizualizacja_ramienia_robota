import pybullet as p
import pybullet_data
import time
import numpy as np
import pygame
from robot_controller import TeachAndPlayController, PickAndPlaceController

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


class RobotSimulation:
    EE_INDEX      = 4   # indeks ogniwa end-effektora w URDF
    NUM_JOINTS    = 4   # liczba sterowanych stawów
    # Ramię jest duże i nie sięga do samej podłogi — końcówka „zawisa" kilka
    # cm nad kostką, dlatego próg chwytania musi to uwzględniać.
    GRAB_THRESHOLD = 0.4  # maksymalny dystans [m] do chwytania kostki

    def __init__(self, gui: bool = True):
        self.gui = gui
        self._setup_physics()
        self._load_models()
        self._setup_ui()
        self._setup_audio()

        self.controller = TeachAndPlayController(self.robot, self.EE_INDEX)
        self.ab_points = {'A': None, 'B': None}
        self.ab_markers = {'A': None, 'B': None}

        self.current_angles   = [0.0] * self.NUM_JOINTS
        self.prev_angles      = [0.0] * self.NUM_JOINTS
        self.prev_slider_vals = [0.3, 0.0, 0.4]
        self.gripper_active   = False
        self.constraint_id    = None
        self.space_pressed    = False
        self.tick_counter     = 0  # do regulacji częstotliwości nagrywania

    # ------------------------------------------------------------------
    # Inicjalizacja
    # ------------------------------------------------------------------

    def _setup_physics(self):
        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def _load_models(self):
        self.plane = p.loadURDF("plane.urdf")
        self.cube  = p.loadURDF("cube.urdf",       [0.45, 0.0, 0.08], useFixedBase=False)
        self.robot = p.loadURDF("simple_arm.urdf", [0,    0,   0   ], useFixedBase=True)

    def _setup_ui(self):
        self.sliders = {
            'x': p.addUserDebugParameter("Ramię Cel X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Ramię Cel Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Ramię Cel Z",  0.1, 1.0, 0.4),
        }
        # Punkt A = pozycja startowa kostki (punkt pobrania).
        # Domyślne wartości leżą w zasięgu ramienia (promień ~0.45 m).
        self.cube_sliders = {
            'x': p.addUserDebugParameter("Punkt A (Kostka) X", -0.8, 0.8, 0.45),
            'y': p.addUserDebugParameter("Punkt A (Kostka) Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Punkt A (Kostka) Z",  0.08, 1.0, 0.08),
        }
        # Punkt B = miejsce, w którym kostka ma zostać odłożona.
        self.point_b_sliders = {
            'x': p.addUserDebugParameter("Punkt B (Cel)  X", -0.8, 0.8, 0.0),
            'y': p.addUserDebugParameter("Punkt B (Cel)  Y", -0.8, 0.8, 0.45),
            'z': p.addUserDebugParameter("Punkt B (Cel)  Z",  0.08, 1.0, 0.08),
        }
        self.buttons = {
            'set_cube': p.addUserDebugParameter("USTAW KOSTKE NA STARCIE",  1, 0, 0),
            'record':   p.addUserDebugParameter(" NAGRYWAJ (START/STOP)", 1, 0, 0),
            'play':     p.addUserDebugParameter(" ODTWORZ SEKWENCJE",     1, 0, 0),
            'export':   p.addUserDebugParameter(" EKSPORTUJ JSON",        1, 0, 0),
            'import':   p.addUserDebugParameter(" IMPORTUJ JSON",         1, 0, 0),
            'save_a':   p.addUserDebugParameter(" ZAPISZ A (KOSTKA)",     1, 0, 0),
            'save_b':   p.addUserDebugParameter(" ZAPISZ B (CEL)",        1, 0, 0),
            'run_ab':   p.addUserDebugParameter(" WYKONAJ A -> B",        1, 0, 0),
            'clear':    p.addUserDebugParameter(" WYCZUSC PAMIEC",        1, 0, 0),
        }
        self.btn_states = {k: 0 for k in self.buttons}

    def _setup_audio(self):
        self.motor_sound = None
        self.motor_channel = None
        try:
            pygame.mixer.init()
            self.motor_sound = pygame.mixer.Sound("motor.wav")
            self.motor_channel = pygame.mixer.Channel(0)
        except FileNotFoundError:
            print(" UWAGA: Brak pliku 'motor.wav'. Dźwięk nie będzie odtwarzany.")
        except pygame.error as exc:
            print(f" UWAGA: Audio niedostępne ({exc}). Dźwięk wyłączony.")

    # ------------------------------------------------------------------
    # Obsługa wejść użytkownika i Audio
    # ------------------------------------------------------------------

    def _check_button(self, btn_name: str) -> bool:
        """Zwraca True dokładnie raz po każdym kliknięciu przycisku."""
        current_clicks = p.readUserDebugParameter(self.buttons[btn_name])
        if current_clicks > self.btn_states[btn_name]:
            self.btn_states[btn_name] = current_clicks
            return True
        return False

    def _clamp_angles(self):
        """Przycina current_angles do limitów zdefiniowanych w JOINT_LIMITS_RAD."""
        for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            self.current_angles[i] = float(np.clip(self.current_angles[i], lo, hi))

    def handle_ik_sliders(self):
        """Sterowanie przez suwaki IK — aktualizuje kąty tylko przy zmianie."""
        vals = [p.readUserDebugParameter(self.sliders[k]) for k in ('x', 'y', 'z')]
        if any(abs(v - pv) > 0.001 for v, pv in zip(vals, self.prev_slider_vals)):
            self.current_angles   = self._solve_ik(vals)
            self.prev_slider_vals = vals

    def handle_keyboard(self):
        """Sterowanie klawiaturą."""
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

        self._clamp_angles()

        # Toggle chwytaka na pierwsze wciśnięcie spacji
        space_down = ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN
        if space_down and not self.space_pressed:
            self._toggle_gripper()
        self.space_pressed = space_down

    def _update_audio(self):
        if not self.motor_sound:
            return

        # Check if any joint is actually moving in the simulation (not just input change)
        is_moving = False
        for i in range(self.NUM_JOINTS):
            joint_state = p.getJointState(self.robot, i)
            joint_velocity = joint_state[1]  # Linear or angular velocity
            if abs(joint_velocity) > 0.01:  # Threshold for detectible motion
                is_moving = True
                break

        if is_moving:
            if not self.motor_channel.get_busy():
                self.motor_channel.play(self.motor_sound, loops=-1)
        else:
            if self.motor_channel.get_busy():
                self.motor_channel.stop()

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
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        ee_pos   = p.getLinkState(self.robot, self.EE_INDEX)[0]

        if np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) >= self.GRAB_THRESHOLD:
            self.gripper_active = False
            return

        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 0)

        ee_pos,   ee_orn   = p.getLinkState(self.robot, self.EE_INDEX)[0:2]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        local_pos, local_orn   = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)

        self.constraint_id = p.createConstraint(
            self.robot, self.EE_INDEX, self.cube, -1,
            p.JOINT_FIXED, [0, 0, 0], local_pos, [0, 0, 0], local_orn
        )

    def _release(self):
        p.removeConstraint(self.constraint_id)
        self.constraint_id = None
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    # ------------------------------------------------------------------
    # Narzędzia
    # ------------------------------------------------------------------

    def _current_target_position(self):
        return [p.readUserDebugParameter(self.sliders[k]) for k in ('x', 'y', 'z')]

    def set_cube_position(self, pos):
        if self.constraint_id is not None:
            self._release()
        self.gripper_active = False
        p.resetBasePositionAndOrientation(self.cube, pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

    def reset_cube_position(self):
        self.set_cube_position([p.readUserDebugParameter(self.cube_sliders[k]) for k in ('x', 'y', 'z')])

    def prepare_cube_for_playback(self):
        if self.controller.waypoints:
            self.set_cube_position(self.controller.waypoints[0]["cube_pos"])
        else:
            self.reset_cube_position()

    def sync_state_after_play(self, playback_constraint=None):
        if not self.controller.waypoints:
            return
        last = self.controller.waypoints[-1]
        self.current_angles = list(last["angles"])
        self.gripper_active  = last["gripper"]
        if playback_constraint is not None:
            self.constraint_id = playback_constraint
        elif not self.gripper_active and self.constraint_id is not None:
            self._release()
        
        # Zapobiega "czknięciu" audio po synchronizacji
        self.prev_angles = list(self.current_angles)

    def export_sequence(self):
        try:
            self.controller.export_sequence()
        except OSError as exc:
            print(f" Nie udalo sie wyeksportowac JSON: {exc}")

    def import_sequence(self):
        try:
            count = self.controller.import_sequence()
        except (OSError, ValueError) as exc:
            print(f" Nie udalo sie zaimportowac JSON: {exc}")
            return

        if count:
            self.prepare_cube_for_playback()

    def save_ab_point(self, name):
        if name == 'A':
            pos = list(p.getBasePositionAndOrientation(self.cube)[0])
        else:
            pos = self._current_target_position()

        self.ab_points[name] = pos
        self._show_ab_marker(name, pos)
        print(f" Zapisano punkt {name}: {[round(v, 3) for v in pos]}")

    def _show_ab_marker(self, name, pos):
        if self.ab_markers[name] is not None:
            p.removeBody(self.ab_markers[name])

        color = [0.1, 0.9, 0.1, 0.75] if name == 'A' else [0.95, 0.25, 0.1, 0.75]
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.045, rgbaColor=color)
        self.ab_markers[name] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=pos)

    def run_ab_program(self):
        try:
            self.controller.build_pick_and_place_sequence(
                self.ab_points['A'],
                self.ab_points['B'],
                self.NUM_JOINTS,
                JOINT_LIMITS_RAD,
            )
        except ValueError as exc:
            print(f" Nie mozna wykonac programu A -> B: {exc}")
            return

        self.set_cube_position(self.ab_points['A'])
        playback_constraint = self.controller.play_sequence(self.cube)
        self.sync_state_after_play(playback_constraint)

    # ------------------------------------------------------------------
    # Główna pętla
    # ------------------------------------------------------------------

    def run(self):
        print("====== SYMULACJA ROBOTA — TRYB NAGRYWANIA ======")
        print("Klawiatura:  baza |  ramię1 | Z/X ramię2 | C/V przegub | SPACJA chwytak")
        print("JSON: EKSPORTUJ JSON / IMPORTUJ JSON używają pliku teach_play_sequence.json")
        print("A -> B: ustaw kostkę i zapisz A, ustaw cel suwakami i zapisz B, potem WYKONAJ A -> B")

        while True:
            self.tick_counter += 1

            # 1. Odczyt wejść
            self.handle_ik_sliders()
            self.handle_keyboard()

            # Aktualizacja dźwięku podczas sterowania ręcznego
            if not self.controller.is_playing:
                self._update_audio()

            # 2. Przyciski UI
            if self._check_button('set_cube'):
                self.reset_cube_position()

            if self._check_button('record'):
                self.controller.toggle_recording()

            if self._check_button('play'):
                self.prepare_cube_for_playback()
                playback_constraint = self.controller.play_sequence(self.cube)
                self.sync_state_after_play(playback_constraint)

            if self._check_button('export'):
                self.export_sequence()

            if self._check_button('import'):
                self.import_sequence()

            if self._check_button('save_a'):
                self.save_ab_point('A')

            if self._check_button('save_b'):
                self.save_ab_point('B')

            if self._check_button('run_ab'):
                self.run_ab_program()

            if self._check_button('clear'):
                self.controller.clear_sequence()

            if self._check_button('save_json'):
                self.controller.save_to_json()

            if self._check_button('load_json'):
                self.controller.load_from_json()

            if self._check_button('pick_place'):
                self.run_pick_and_place()

            # 3. Ciągłe nagrywanie w tle
            if self.controller.is_recording and self.tick_counter % 10 == 0:
                self.controller.record_frame(self.current_angles, self.gripper_active, self.cube)

            # 4. Fizyka
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
