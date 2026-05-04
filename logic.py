import pybullet as p
import pybullet_data
import time
import numpy as np # Upewnij się, że masz numpy (pip install numpy)

from robot_controller import TeachAndPlayController

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
cubeId = p.loadURDF("cube.urdf", [0.3, 0.0, 0.2], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
robotId = p.loadURDF("simple_arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

endEffectorIndex = 4 

controller = TeachAndPlayController(robotId, endEffectorIndex)

slider_x = p.addUserDebugParameter("Cel X (Suwak)", -0.8, 0.8, 0.3)
slider_y = p.addUserDebugParameter("Cel Y (Suwak)", -0.8, 0.8, 0.0)
slider_z = p.addUserDebugParameter("Cel Z (Suwak)", 0.1, 1.0, 0.4)

btn_save = p.addUserDebugParameter("ZAPISZ PUNKT", 1, 0, 0)
btn_play = p.addUserDebugParameter("ODTWORZ SEKWENCJE", 1, 0, 0)
btn_clear = p.addUserDebugParameter("WYCZYSC PAMIEC", 1, 0, 0)

prev_save_clicks, prev_play_clicks, prev_clear_clicks = 0, 0, 0
current_joint_angles = [0.0, 0.0, 0.0, 0.0]
prev_slider_vals = [0.3, 0.0, 0.4]

gripper_active = False
constraint_id = None
space_pressed_previously = False

print("========================================")
print("STEROWANIE GOTOWE")
print("Zwróć uwagę na konsolę przy zapisywaniu punktów.")
print("========================================")

while True:
    slider_vals = [
        p.readUserDebugParameter(slider_x),
        p.readUserDebugParameter(slider_y),
        p.readUserDebugParameter(slider_z)
    ]
    
    if abs(slider_vals[0] - prev_slider_vals[0]) > 0.001 or \
       abs(slider_vals[1] - prev_slider_vals[1]) > 0.001 or \
       abs(slider_vals[2] - prev_slider_vals[2]) > 0.001:
        
        ik_angles = p.calculateInverseKinematics(robotId, endEffectorIndex, slider_vals)
        for i in range(len(current_joint_angles)):
            if i < len(ik_angles):
                current_joint_angles[i] = ik_angles[i]
        prev_slider_vals = slider_vals.copy()

    keys = p.getKeyboardEvents()
    delta = 0.01 # Zmniejszyłem lekko deltę dla płynniejszego ręcznego ruchu

    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        current_joint_angles[0] -= delta
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        current_joint_angles[0] += delta
        
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        current_joint_angles[1] -= delta
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        current_joint_angles[1] += delta
        
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
        current_joint_angles[2] += delta
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        current_joint_angles[2] -= delta

    if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
        current_joint_angles[3] += delta
    if ord('v') in keys and keys[ord('v')] & p.KEY_IS_DOWN:
        current_joint_angles[3] -= delta

    # --- OBSŁUGA CHWYTAKA W TRAKCIE RĘCZNEGO UCZENIA ---
    space_is_down = (ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN)
    
    if space_is_down and not space_pressed_previously:
        gripper_active = not gripper_active
        
        if gripper_active and constraint_id is None:
            cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
            ee_state = p.getLinkState(robotId, endEffectorIndex)
            ee_pos = ee_state[0]
            dist = sum([(a - b)**2 for a, b in zip(cube_pos, ee_pos)])**0.5
            
            if dist < 0.15:
                # Wyłączamy kolizje z całym robotem
                for i in range(-1, p.getNumJoints(robotId)):
                    p.setCollisionFilterPair(robotId, cubeId, i, -1, enableCollision=0)
                
                # NOWE: Obliczamy dokładną pozycję kostki względem chwytaka (aby uniknąć szarpania)
                ee_pos, ee_orn = ee_state[0], ee_state[1] # ee_state pobraliśmy kilka linijek wyżej
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cubeId)
                
                # Matematyka transformacji przestrzennych (z globalnych na lokalne chwytaka)
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
                local_cube_pos, local_cube_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, cube_pos, cube_orn)
                
                # Tworzymy "spaw" uwzględniający to przesunięcie
                constraint_id = p.createConstraint(parentBodyUniqueId=robotId,
                                                   parentLinkIndex=endEffectorIndex,
                                                   childBodyUniqueId=cubeId,
                                                   childLinkIndex=-1,
                                                   jointType=p.JOINT_FIXED,
                                                   jointAxis=[0, 0, 0],
                                                   parentFramePosition=local_cube_pos,
                                                   childFramePosition=[0, 0, 0],
                                                   parentFrameOrientation=local_cube_orn)
                print("--- ZŁAPANO KOSTKĘ BEZ NAPRĘŻEŃ ---")
            else:
                gripper_active = False
                print("Za daleko od kostki!")
                
        elif not gripper_active and constraint_id is not None:
            p.removeConstraint(constraint_id)
            constraint_id = None
            # PRZYWRACAMY KOLIZJE
            p.setCollisionFilterPair(robotId, cubeId, endEffectorIndex, -1, enableCollision=1)
            print("--- PUSZCZONO KOSTKĘ ---")
            
    space_pressed_previously = space_is_down

    save_clicks = p.readUserDebugParameter(btn_save)
    play_clicks = p.readUserDebugParameter(btn_play)
    clear_clicks = p.readUserDebugParameter(btn_clear)

    if save_clicks > prev_save_clicks:
        # PRZEKAZUJEMY cubeId ABY ZAPISAĆ JEJ POZYCJĘ
        controller.save_waypoint(current_joint_angles, gripper_active, cubeId)
        prev_save_clicks = save_clicks

    if play_clicks > prev_play_clicks:
        controller.play_sequence(cubeId)
        
        if controller.waypoints:
            last_step = controller.waypoints[-1]
            current_joint_angles = list(last_step["angles"])
            gripper_active = last_step["gripper"]
            if not gripper_active and constraint_id is not None:
                p.removeConstraint(constraint_id)
                constraint_id = None
                p.setCollisionFilterPair(robotId, cubeId, endEffectorIndex, -1, enableCollision=1)
        prev_play_clicks = play_clicks

    if clear_clicks > prev_clear_clicks:
        controller.clear_sequence()
        prev_clear_clicks = clear_clicks

    if not controller.is_playing:
        for i in range(len(current_joint_angles)):
            p.setJointMotorControl2(bodyIndex=robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=current_joint_angles[i],
                                    force=200,
                                    maxVelocity=1.5) # Ograniczenie prędkości również przy ręcznym sterowaniu

    p.stepSimulation()
    time.sleep(1./240.)