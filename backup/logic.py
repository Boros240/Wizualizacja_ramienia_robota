import pybullet as p
import pybullet_data
import time

# IMPORTUJEMY NASZĄ NOWĄ KLASĘ Z DRUGIEGO PLIKU
from robot_controller import TeachAndPlayController

# 1. Uruchomienie PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Załadowanie otoczenia i robota
planeId = p.loadURDF("plane.urdf")
cubeId = p.loadURDF("cube.urdf", [0.5, 0, 0.5], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
robotId = p.loadURDF("simple_arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
endEffectorIndex = 3

# INICJALIZACJA NASZEGO OBIEKTU TEACH & PLAY
controller = TeachAndPlayController(robotId, endEffectorIndex)

# 3. Suwaki do sterowania ręcznego
slider_x = p.addUserDebugParameter("Cel X", -0.8, 0.8, 0.3)
slider_y = p.addUserDebugParameter("Cel Y", -0.8, 0.8, 0.0)
slider_z = p.addUserDebugParameter("Cel Z", 0.1, 1.0, 0.5)
slider_alfa = p.addUserDebugParameter("Cel Alfa", -3.14, 3.14, 0)
# Tworzymy czerwoną kulkę
visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
target_marker = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=[0.3, 0, 0.5])

# 4. Przyciski UI - "Hak" w PyBullet: Jeśli min=1, max=0, start=0 to tworzy się PRZYCISK
btn_save = p.addUserDebugParameter("ZAPISZ PUNKT", 1, 0, 0)
btn_play = p.addUserDebugParameter("ODTWORZ SEKWENCJE", 1, 0, 0)
btn_clear = p.addUserDebugParameter("WYCZYSC PAMIEC", 1, 0, 0)

# Zmienne do śledzenia kliknięć (PyBullet zwraca ilość kliknięć od początku uruchomienia)
prev_save_clicks = 0
prev_play_clicks = 0
prev_clear_clicks = 0

print("Ustaw ramię suwakami i kliknij 'ZAPISZ PUNKT'.")

# 5. Główna pętla
while True:
    # Odczyt z suwaków (sterowanie ręczne)
    target_x = p.readUserDebugParameter(slider_x)
    target_y = p.readUserDebugParameter(slider_y)
    target_z = p.readUserDebugParameter(slider_z)
    target_alfa = p.readUserDebugParameter(slider_alfa)
    target_pos = [target_x, target_y, target_z]
    
    # Odczyt ilości kliknięć w przyciski
    save_clicks = p.readUserDebugParameter(btn_save)
    play_clicks = p.readUserDebugParameter(btn_play)
    clear_clicks = p.readUserDebugParameter(btn_clear)

    # Sprawdzamy czy przycisk ZAPISZ został kliknięty
    if save_clicks > prev_save_clicks:
        controller.save_waypoint(target_pos)
        prev_save_clicks = save_clicks

    # Sprawdzamy czy przycisk ODTWÓRZ został kliknięty
    if play_clicks > prev_play_clicks:
        # Odpalamy metodę z naszego obiektu
        controller.play_sequence(target_marker)
        prev_play_clicks = play_clicks
        
    # Sprawdzamy czy przycisk WYCZYŚĆ został kliknięty
    if clear_clicks > prev_clear_clicks:
        controller.clear_sequence()
        prev_clear_clicks = clear_clicks

    # Jeśli nie odtwarzamy, wykonujemy normalne sterowanie suwakami
    p.resetBasePositionAndOrientation(target_marker, target_pos, [0, 0, 0, 1])
    joint_angles = p.calculateInverseKinematics(robotId, endEffectorIndex, target_pos)
    for i in range(len(joint_angles)):
        p.setJointMotorControl2(bodyIndex=robotId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_angles[i],
                                force=100)
    
    p.stepSimulation()
    time.sleep(1./240.)