import pybullet as p
import pybullet_data
import time

from robot_controller import TeachAndPlayController

# 1. Uruchomienie PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Załadowanie otoczenia, robota i KOSTKI
planeId = p.loadURDF("plane.urdf")
cubeId = p.loadURDF("cube.urdf", [0.3, 0.0, 0.2], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
robotId = p.loadURDF("simple_arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

# UWAGA: Ustaw poprawny indeks końcówki! (zależnie od tego, jak dokładnie nazwałeś link w URDF)
endEffectorIndex = 4 

controller = TeachAndPlayController(robotId, endEffectorIndex)

# 3. Suwaki (Kinematyka Odwrotna - wprowadzanie liczbowe)
slider_x = p.addUserDebugParameter("Cel X (Suwak)", -0.8, 0.8, 0.3)
slider_y = p.addUserDebugParameter("Cel Y (Suwak)", -0.8, 0.8, 0.0)
slider_z = p.addUserDebugParameter("Cel Z (Suwak)", 0.1, 1.0, 0.4)

# 4. Przyciski Teach & Play
btn_save = p.addUserDebugParameter("ZAPISZ PUNKT", 1, 0, 0)
btn_play = p.addUserDebugParameter("ODTWORZ SEKWENCJE", 1, 0, 0)
btn_clear = p.addUserDebugParameter("WYCZYSC PAMIEC", 1, 0, 0)

prev_save_clicks, prev_play_clicks, prev_clear_clicks = 0, 0, 0

# 5. Stan początkowy
current_joint_angles = [0.0, 0.0, 0.0, 0.0] # 4 złącza, jeśli masz nadgarstek
prev_slider_vals = [0.3, 0.0, 0.4]

# Zmienne chwytaka
gripper_active = False
constraint_id = None
space_pressed_previously = False # do wykrywania pojedynczego wciśnięcia spacji

print("========================================")
print("STEROWANIE:")
print("Suwaki (X,Y,Z): Ruch do wybranego punktu")
print("Strzałki: Obrót podstawy i dolnego ramienia")
print("Klawisze Z / X: Górne ramię")
print("Klawisze C / V: Nadgarstek/Końcówka")
print("SPACJA: Złap / Puść kostkę")
print("========================================")

while True:
    # --- 1. ODCZYT SUWAKÓW (Wprowadzanie położenia) ---
    slider_vals = [
        p.readUserDebugParameter(slider_x),
        p.readUserDebugParameter(slider_y),
        p.readUserDebugParameter(slider_z)
    ]
    
    # Jeśli ruszyliśmy suwakiem, nadpisujemy kąty za pomocą Kinematyki Odwrotnej
    if abs(slider_vals[0] - prev_slider_vals[0]) > 0.001 or \
       abs(slider_vals[1] - prev_slider_vals[1]) > 0.001 or \
       abs(slider_vals[2] - prev_slider_vals[2]) > 0.001:
        
        ik_angles = p.calculateInverseKinematics(robotId, endEffectorIndex, slider_vals)
        for i in range(len(current_joint_angles)):
            if i < len(ik_angles):
                current_joint_angles[i] = ik_angles[i]
        prev_slider_vals = slider_vals.copy()

    # --- 2. ODCZYT KLAWIATURY (Kinematyka Prosta - złącza bezpośrednio) ---
    keys = p.getKeyboardEvents()
    delta = 0.02 # Szybkość obrotu klawiszami

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

    # Opcjonalne: jeśli masz przegub nadgarstka
    if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
        current_joint_angles[3] += delta
    if ord('v') in keys and keys[ord('v')] & p.KEY_IS_DOWN:
        current_joint_angles[3] -= delta

    # --- 3. OBSŁUGA CHWYTAKA (SPACJA) ---
    space_is_down = (ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN)
    
    if space_is_down and not space_pressed_previously:
        gripper_active = not gripper_active # Przełącz stan
        
        if gripper_active and constraint_id is None:
            # Sprawdzamy, czy jesteśmy blisko kostki, żeby ją złapać
            cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
            ee_state = p.getLinkState(robotId, endEffectorIndex)
            ee_pos = ee_state[0]
            dist = sum([(a - b)**2 for a, b in zip(cube_pos, ee_pos)])**0.5
            
            if dist < 0.15:
                constraint_id = p.createConstraint(robotId, endEffectorIndex, cubeId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
                print("--- ZŁAPANO KOSTKĘ ---")
            else:
                gripper_active = False # Zbyt daleko, cofamy chęć złapania
                print(f"Jesteś za daleko od kostki! (Dystans: {dist:.2f}, wymagane: < 0.15)")
                
        elif not gripper_active and constraint_id is not None:
            p.removeConstraint(constraint_id)
            constraint_id = None
            print("--- PUSZCZONO KOSTKĘ ---")
            
    space_pressed_previously = space_is_down

    # --- 4. TEACH & PLAY PRZYCISKI ---
    save_clicks = p.readUserDebugParameter(btn_save)
    play_clicks = p.readUserDebugParameter(btn_play)
    clear_clicks = p.readUserDebugParameter(btn_clear)

    if save_clicks > prev_save_clicks:
        controller.save_waypoint(current_joint_angles, gripper_active)
        prev_save_clicks = save_clicks

    if play_clicks > prev_play_clicks:
        controller.play_sequence(cubeId)
        # Po zakończeniu odtwarzania aktualizujemy nasze zmienne do stanu końcowego
        if controller.waypoints:
            last_step = controller.waypoints[-1]
            current_joint_angles = list(last_step["angles"])
            gripper_active = last_step["gripper"]
            # Wymuszamy aktualizację chwytaka, żeby stan UI się zgadzał z fizyką
            if not gripper_active and constraint_id is not None:
                p.removeConstraint(constraint_id)
                constraint_id = None
        prev_play_clicks = play_clicks

    if clear_clicks > prev_clear_clicks:
        controller.clear_sequence()
        prev_clear_clicks = clear_clicks

    # --- 5. RUCH SILNIKÓW ---
    if not controller.is_playing:
        for i in range(len(current_joint_angles)):
            p.setJointMotorControl2(bodyIndex=robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=current_joint_angles[i],
                                    force=200)

    p.stepSimulation()
    time.sleep(1./240.)