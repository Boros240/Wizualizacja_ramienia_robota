import pybullet as p
import pybullet_data
import time
import numpy as np
from robot_controller import TeachAndPlayController

class RobotSimulation:
    def __init__(self):
        self.EE_INDEX = 4
        self.NUM_JOINTS = 4
        self.GRAB_THRESHOLD = 0.15
        
        self._setup_physics()
        self._load_models()
        self._setup_ui()
        
        self.controller = TeachAndPlayController(self.robot, self.EE_INDEX)
        
        self.current_angles = [0.0] * self.NUM_JOINTS
        self.prev_slider_vals = [0.3, 0.0, 0.4]
        self.gripper_active = False
        self.constraint_id = None
        self.space_pressed = False
        
        # NOWE: Licznik pętli do optymalizacji częstotliwości nagrywania
        self.tick_counter = 0 

    def _setup_physics(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def _load_models(self):
        self.plane = p.loadURDF("plane.urdf")
        self.cube = p.loadURDF("cube.urdf", [0.3, 0.0, 0.2], useFixedBase=False)
        self.robot = p.loadURDF("simple_arm.urdf", [0, 0, 0], useFixedBase=True)

    def _setup_ui(self):
        self.sliders = {
            'x': p.addUserDebugParameter("Ramie Cel X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Ramie Cel Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Ramie Cel Z", 0.1, 1.0, 0.4),
        }
        self.cube_sliders = {
            'x': p.addUserDebugParameter("Start Kostki X", -0.8, 0.8, 0.3),
            'y': p.addUserDebugParameter("Start Kostki Y", -0.8, 0.8, 0.0),
            'z': p.addUserDebugParameter("Start Kostki Z", 0.05, 1.0, 0.2),
        }
        self.buttons = {
            'set_cube': p.addUserDebugParameter("USTAW KOSTKE NA STARCIE", 1, 0, 0),
            'record': p.addUserDebugParameter(" NAGRYWAJ (START/STOP)", 1, 0, 0), # ZMIENIONO PRZYCISK
            'play': p.addUserDebugParameter(" ODTWORZ SEKWENCJE", 1, 0, 0),
            'clear': p.addUserDebugParameter(" WYCZYSC PAMIEC", 1, 0, 0),
        }
        self.btn_states = {k: 0 for k in self.buttons}

    def _check_button(self, btn_name: str) -> bool:
        current_clicks = p.readUserDebugParameter(self.buttons[btn_name])
        if current_clicks > self.btn_states[btn_name]:
            self.btn_states[btn_name] = current_clicks
            return True
        return False

    def handle_ik_sliders(self):
        vals = [p.readUserDebugParameter(self.sliders[k]) for k in ['x', 'y', 'z']]
        if any(abs(v - pv) > 0.001 for v, pv in zip(vals, self.prev_slider_vals)):
            ik_angles = p.calculateInverseKinematics(self.robot, self.EE_INDEX, vals)
            self.current_angles = list(ik_angles)[:self.NUM_JOINTS]
            self.prev_slider_vals = vals

    def handle_keyboard(self):
        keys = p.getKeyboardEvents()
        delta = 0.01 
        
        key_map = {
            p.B3G_LEFT_ARROW: (0, -delta), p.B3G_RIGHT_ARROW: (0, delta),
            p.B3G_UP_ARROW: (1, -delta),   p.B3G_DOWN_ARROW: (1, delta),
            ord('z'): (2, delta),          ord('x'): (2, -delta),
            ord('c'): (3, delta),          ord('v'): (3, -delta)
        }
        
        for key, (joint_idx, d) in key_map.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                self.current_angles[joint_idx] += d

        space_down = ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN
        if space_down and not self.space_pressed:
            self._toggle_gripper()
        self.space_pressed = space_down

    def _toggle_gripper(self):
        self.gripper_active = not self.gripper_active
        
        if self.gripper_active and self.constraint_id is None:
            cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
            ee_pos = p.getLinkState(self.robot, self.EE_INDEX)[0]
            
            if np.linalg.norm(np.array(cube_pos) - np.array(ee_pos)) < self.GRAB_THRESHOLD:
                for i in range(-1, p.getNumJoints(self.robot)):
                    p.setCollisionFilterPair(self.robot, self.cube, i, -1, 0)
                
                ee_pos, ee_orn = p.getLinkState(self.robot, self.EE_INDEX)[0:2]
                cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
                local_cube_pos, local_cube_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)
                
                self.constraint_id = p.createConstraint(self.robot, self.EE_INDEX, self.cube, -1, p.JOINT_FIXED, [0, 0, 0], local_cube_pos, [0, 0, 0], local_cube_orn)
            else:
                self.gripper_active = False
                
        elif not self.gripper_active and self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None
            for i in range(-1, p.getNumJoints(self.robot)):
                p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    def reset_cube_position(self):
        cx, cy, cz = [p.readUserDebugParameter(self.cube_sliders[k]) for k in ['x', 'y', 'z']]
        p.resetBasePositionAndOrientation(self.cube, [cx, cy, cz], [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0])

    def sync_state_after_play(self):
        if self.controller.waypoints:
            last = self.controller.waypoints[-1]
            self.current_angles = list(last["angles"])
            self.gripper_active = last["gripper"]
            if not self.gripper_active and self.constraint_id is not None:
                p.removeConstraint(self.constraint_id)
                self.constraint_id = None
                for i in range(-1, p.getNumJoints(self.robot)):
                    p.setCollisionFilterPair(self.robot, self.cube, i, -1, 1)

    def run(self):
        print("====== SYMULACJA CIĄGŁEGO NAGRYWANIA ======")
        while True:
            self.tick_counter += 1
            
            # 1. Odczyt użytkownika
            self.handle_ik_sliders()
            self.handle_keyboard()

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

            # 3. ZAPIS CIĄGŁY W TLE
            # Zapisujemy stan co 10 "tyknięć" (czyli 240Hz / 10 = 24 razy na sekundę)
            if self.controller.is_recording and self.tick_counter % 10 == 0:
                self.controller.record_frame(self.current_angles, self.gripper_active, self.cube)

            # 4. Fizyka
            if not self.controller.is_playing:
                for i, angle in enumerate(self.current_angles):
                    p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=angle, force=200, maxVelocity=1.5)

            p.stepSimulation()
            time.sleep(1./240.)

if __name__ == "__main__":
    app = RobotSimulation()
    app.run()