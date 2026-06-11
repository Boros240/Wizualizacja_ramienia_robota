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
    EE_INDEX      = 4   # ogniwo chwytaka (do więzu trzymającego kostkę)
    TCP_INDEX     = 5   # wirtualny punkt narzędziowy MIĘDZY palcami (IK + dystans)
    NUM_JOINTS    = 4   # liczba sterowanych stawów
    # IK celuje teraz TCP wprost w środek kostki, więc dystans chwytania jest
    # mały — wystarczy niewielki próg (kostka faktycznie jest między palcami).
    GRAB_THRESHOLD = 0.1  # maksymalny dystans [m] do chwytania kostki

    def __init__(self, gui: bool = True):
        self.gui = gui
        self._setup_physics()
        self._load_models()
        self._setup_ui()
        self._setup_audio()

        self.controller = TeachAndPlayController(self.robot, self.EE_INDEX, self.TCP_INDEX)
        self.pick_and_place = PickAndPlaceController(
            move_fn=self.move_ee_to,
            grab_fn=self._auto_grab,
            release_fn=self._auto_release,
        )

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
        self.cube  = p.loadURDF("cube.urdf",       [0.45, 0.0, 0.06], useFixedBase=False)
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
            'z': p.addUserDebugParameter("Punkt A (Kostka) Z",  0.06, 1.0, 0.06),
        }
        # Punkt B = miejsce, w którym kostka ma zostać odłożona.
        self.point_b_sliders = {
            'x': p.addUserDebugParameter("Punkt B (Cel)  X", -0.8, 0.8, 0.0),
            'y': p.addUserDebugParameter("Punkt B (Cel)  Y", -0.8, 0.8, 0.45),
            'z': p.addUserDebugParameter("Punkt B (Cel)  Z",  0.06, 1.0, 0.06),
        }
        self.buttons = {
            'set_cube':   p.addUserDebugParameter("USTAW KOSTKE NA STARCIE",  1, 0, 0),
            'record':     p.addUserDebugParameter(" NAGRYWAJ (START/STOP)", 1, 0, 0),
            'play':       p.addUserDebugParameter(" ODTWORZ SEKWENCJE",     1, 0, 0),
            'clear':      p.addUserDebugParameter(" WYCZUSC PAMIEC",        1, 0, 0),
            'save_json':  p.addUserDebugParameter(" ZAPISZ DO JSON",        1, 0, 0),
            'load_json':  p.addUserDebugParameter(" WCZYTAJ Z JSON",        1, 0, 0),
            'pick_place': p.addUserDebugParameter(" WYKONAJ A->B (PICK&PLACE)", 1, 0, 0),
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

    def _disable_cube_collision(self):
        """Wyłącza kolizje robot↔kostka (by chwytak nie odpychał kostki)."""
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 0)

    def _enable_cube_collision(self):
        """Przywraca kolizje robot↔kostka."""
        for i in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    def _set_gripper_floor_collision(self, enabled: int):
        """Włącza/wyłącza kolizję palców z podłożem.

        Aby chwycić kostkę leżącą na podłodze, palce muszą zejść do jej
        poziomu — wtedy stykają się z podłożem i blokują ruch. Na czas
        sekwencji Pick & Place wyłączamy tę kolizję.
        """
        for li in (self.EE_INDEX, self.TCP_INDEX):
            p.setCollisionFilterPair(self.robot, self.plane, li, -1, enabled)

    def _try_grab(self):
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        tcp_pos  = p.getLinkState(self.robot, self.TCP_INDEX)[4]

        if np.linalg.norm(np.array(cube_pos) - np.array(tcp_pos)) >= self.GRAB_THRESHOLD:
            self.gripper_active = False
            return

        self._disable_cube_collision()

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
        self._enable_cube_collision()
        self._set_gripper_floor_collision(1)

    # ------------------------------------------------------------------
    # Narzędzia
    # ------------------------------------------------------------------

    def reset_cube_position(self):
        cx, cy, cz = [p.readUserDebugParameter(self.cube_sliders[k]) for k in ('x', 'y', 'z')]
        p.resetBasePositionAndOrientation(self.cube, [cx, cy, cz], [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

    def _read_point(self, sliders: dict) -> list:
        """Odczytuje pozycję XYZ z trzech suwaków."""
        return [p.readUserDebugParameter(sliders[k]) for k in ('x', 'y', 'z')]

    # ------------------------------------------------------------------
    # Ruch IK i chwytak dla trybu Pick & Place
    # ------------------------------------------------------------------

    def _solve_ik(self, target_pos) -> list:
        """Wyznacza kąty stawów dla zadanej pozycji końcówki.

        W przeciwieństwie do wywołania domyślnego, przekazujemy limity
        stawów, ich zakresy oraz bieżącą pozę jako „rest pose". Bez tego
        solver zwracał kąty poza zakresem ruchu, przez co ramię nigdy nie
        docierało do celu.
        """
        lower = [lo for lo, hi in JOINT_LIMITS_RAD]
        upper = [hi for lo, hi in JOINT_LIMITS_RAD]
        ranges = [hi - lo for lo, hi in JOINT_LIMITS_RAD]
        ik_angles = p.calculateInverseKinematics(
            self.robot, self.TCP_INDEX, target_pos,
            lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
            restPoses=list(self.current_angles),
            maxNumIterations=300, residualThreshold=1e-5,
        )
        return [
            float(np.clip(a, lo, hi))
            for a, (lo, hi) in zip(list(ik_angles)[:self.NUM_JOINTS], JOINT_LIMITS_RAD)
        ]

    def move_ee_to(self, target_pos, max_steps: int = 900,
                   tol: float = 0.015, vel_eps: float = 0.03):
        """Przesuwa końcówkę robota do zadanej pozycji XYZ.

        Ruch kończy się, gdy spełniony jest KTÓRYKOLWIEK warunek:
          • ramię osiągnęło zadane kąty (błąd < `tol`), albo
          • ramię się ZATRZYMAŁO (prędkości stawów ~0 przez kilka kroków) —
            co oznacza, że dalej już nie dojedzie (cel nieosiągalny/limit),
          • przekroczono twardy limit `max_steps`.

        Wcześniej ruch zawsze mielił do `max_steps` (≈10 s w GUI), przez co
        sekwencja A→B sprawiała wrażenie zawieszonej i nie dochodziła do
        etapu puszczenia kostki. Warunek prędkościowy kończy każdy ruch
        natychmiast po ustabilizowaniu się ramienia.
        """
        target_angles = self._solve_ik(target_pos)
        settled = 0

        for _ in range(max_steps):
            for i in range(self.NUM_JOINTS):
                p.setJointMotorControl2(
                    self.robot, i,
                    p.POSITION_CONTROL,
                    targetPosition=target_angles[i],
                    force=400,
                    maxVelocity=2.5,
                )
            states = [p.getJointState(self.robot, i) for i in range(self.NUM_JOINTS)]
            pos_err = max(abs(states[i][0] - target_angles[i]) for i in range(self.NUM_JOINTS))
            vel_max = max(abs(states[i][1]) for i in range(self.NUM_JOINTS))
            self._update_audio()
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

            if pos_err < tol:
                break
            # Ramię stanęło, choć nie dotarło do celu — nie ma sensu czekać.
            settled = settled + 1 if vel_max < vel_eps else 0
            if settled >= 25:
                break

        self.current_angles = [p.getJointState(self.robot, i)[0] for i in range(self.NUM_JOINTS)]
        self.prev_angles = list(self.current_angles)

    def _auto_grab(self):
        """Chwyt kostki używany przez sekwencję Pick & Place."""
        self.gripper_active = True
        if self.constraint_id is None:
            self._try_grab()

    def _auto_release(self):
        """Puszczenie kostki w sekwencji Pick & Place.

        Usuwa TYLKO więz — kolizje pozostają wyłączone, aż ramię odsunie się
        w górę (przywraca je `run_pick_and_place`). Dzięki temu palce nie
        odpychają kostki w momencie puszczenia.
        """
        self.gripper_active = False
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None

    def run_pick_and_place(self):
        """Ustawia punkty A/B z suwaków i uruchamia sekwencję pobierz-i-odłóż."""
        point_a = self._read_point(self.cube_sliders)
        point_b = self._read_point(self.point_b_sliders)
        self.reset_cube_position()
        # Na czas sekwencji wyłączamy kolizje robot↔kostka (by chwytak nie
        # strącił kostki) oraz palce↔podłoga (by palce mogły zejść do kostki
        # leżącej na podłodze). Zostaną przywrócone przy puszczeniu.
        self._disable_cube_collision()
        self._set_gripper_floor_collision(0)
        self.pick_and_place.set_points(point_a, point_b)
        self.pick_and_place.execute()
        # Ramię jest już odsunięte w górę — bezpiecznie przywracamy kolizje
        # (gdyby coś jeszcze trzymało kostkę, najpierw ją puszczamy).
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None
        self._enable_cube_collision()
        self._set_gripper_floor_collision(1)

    def sync_state_after_play(self):
        if not self.controller.waypoints:
            return
        last = self.controller.waypoints[-1]
        self.current_angles = list(last["angles"])
        self.gripper_active  = last["gripper"]
        if not self.gripper_active and self.constraint_id is not None:
            self._release()
        
        # Zapobiega "czknięciu" audio po synchronizacji
        self.prev_angles = list(self.current_angles)

    # ------------------------------------------------------------------
    # Główna pętla
    # ------------------------------------------------------------------

    def run(self):
        print("====== SYMULACJA ROBOTA — TRYB NAGRYWANIA ======")
        print("Klawiatura:  baza |  ramię1 | Z/X ramię2 | C/V przegub | SPACJA chwytak")
        print("Przyciski:   NAGRYWAJ / ODTWORZ / ZAPISZ JSON / WCZYTAJ JSON")
        print("Pick & Place: ustaw punkt A (kostka) i B (cel), naciśnij WYKONAJ A->B")

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
                self.reset_cube_position()
                self.controller.play_sequence(self.cube)
                self.sync_state_after_play()

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
