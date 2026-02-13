#!/usr/bin/env python3
"""
PyBullet Rendering & Physics Benchmark

Benchmarks PyBullet rendering and physics performance without any VLA/torch/lerobot
dependencies. Useful for comparing WSL vs Windows performance.

Dependencies: pip install pybullet numpy pillow
"""

import argparse
import numpy as np
from PIL import Image
import time
import pybullet as p
import pybullet_data
import signal
import sys
import platform

# Parse command line arguments
parser = argparse.ArgumentParser(description='PyBullet Rendering & Physics Benchmark')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps (default: 100)')
parser.add_argument('--single-camera', action='store_true', help='Use only one camera view')
parser.add_argument('--headless', action='store_true', help='Run without GUI (faster, no visualization)')
parser.add_argument('--renderer', type=str, default='tiny', choices=['tiny', 'opengl'],
                    help='Renderer: tiny (CPU, default) or opengl (GPU if available)')
args = parser.parse_args()

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('\n\nInterrupted. Cleaning up...')
    try:
        p.disconnect()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Print system info
print("=" * 60)
print("PyBullet Rendering & Physics Benchmark")
print("=" * 60)
print(f"  Platform: {platform.system()} {platform.release()}")
print(f"  Python:   {platform.python_version()}")
print(f"  Renderer: {'OpenGL (GPU)' if args.renderer == 'opengl' else 'TinyRenderer (CPU)'}")
print(f"  Cameras:  {'1 (single)' if args.single_camera else '3 (all views)'}")
print(f"  GUI:      {'off (headless)' if args.headless else 'on (512x512)'}")
print(f"  Steps:    {args.steps}")
print("=" * 60)

# Initialize PyBullet
print("\nInitializing PyBullet...")
if args.headless:
    physics_client = p.connect(p.DIRECT)
    GUI_MODE = False
else:
    GUI_WIDTH, GUI_HEIGHT = 512, 512
    physics_client = p.connect(p.GUI, options=f"--width={GUI_WIDTH} --height={GUI_HEIGHT}")
    GUI_MODE = True

# Set up simulation
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240.)

if GUI_MODE:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

# Load scene
plane_id = p.loadURDF("plane.urdf")

print("Loading SO-101 robot from URDF...")
robot_urdf_path = "assets/so101/so101.urdf"
robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0], useFixedBase=True)

num_joints = p.getNumJoints(robot_id)
controllable_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(i)
num_dof = min(len(controllable_joints), 6)

# Create cubes (matching MuJoCo scene)
red_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
red_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=[1, 0, 0, 1])
red_cube_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=red_col,
                                 baseVisualShapeIndex=red_vis, basePosition=[0.25, 0.0, 0.03])

blue_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
blue_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=[0, 0, 1, 1])
blue_cube_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=blue_col,
                                  baseVisualShapeIndex=blue_vis, basePosition=[0.15, 0.15, 0.03])

# Camera setup
VLA_WIDTH, VLA_HEIGHT = 256, 256

camera_configs = {
    'third_person': {
        'target': [0.2, 0.0, 0.2],
        'distance': 0.6,
        'yaw': 35,
        'pitch': -25,
    },
    'top_down': {
        'target': [0.2, 0.0, 0.0],
        'distance': 0.8,
        'yaw': 0,
        'pitch': -89,
    },
    'wrist_cam': {
        'target': [0.25, 0.0, 0.1],
        'distance': 0.25,
        'yaw': 90,
        'pitch': -35,
    }
}

if GUI_MODE:
    tp = camera_configs['third_person']
    p.resetDebugVisualizerCamera(
        cameraDistance=tp['distance'], cameraYaw=tp['yaw'],
        cameraPitch=tp['pitch'], cameraTargetPosition=tp['target']
    )

def render_camera(camera_name, renderer_type='tiny'):
    """Render a camera view. Returns (rgb_array, getCameraImage_time)."""
    # GUI shared rendering for third_person
    if GUI_MODE and camera_name == 'third_person':
        t0 = time.time()
        w, h, rgb, _, _ = p.getCameraImage(
            GUI_WIDTH, GUI_HEIGHT,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if renderer_type == 'opengl' else p.ER_TINY_RENDERER
        )
        get_image_time = time.time() - t0

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(GUI_HEIGHT, GUI_WIDTH, 4)[:, :, :3]
        pil_img = Image.fromarray(rgb_array)
        pil_img = pil_img.resize((VLA_WIDTH, VLA_HEIGHT), Image.Resampling.BILINEAR)
        return np.array(pil_img), get_image_time

    config = camera_configs[camera_name]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=config['target'], distance=config['distance'],
        yaw=config['yaw'], pitch=config['pitch'], roll=0, upAxisIndex=2
    )
    projection_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 5.0)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if renderer_type == 'opengl' else p.ER_TINY_RENDERER

    t0 = time.time()
    w, h, rgb, _, _ = p.getCameraImage(VLA_WIDTH, VLA_HEIGHT, view_matrix, projection_matrix, renderer=renderer)
    get_image_time = time.time() - t0

    rgb_array = np.array(rgb, dtype=np.uint8).reshape(VLA_HEIGHT, VLA_WIDTH, 4)[:, :, :3]
    return rgb_array, get_image_time

# Settle physics
print("Settling physics...")
for _ in range(240):
    p.stepSimulation()

# Main benchmark loop
print(f"\nRunning {args.steps} steps...")
profile_data = {
    'render': [],
    'get_image': [],
    'physics': [],
    'total': []
}

for step in range(args.steps):
    iter_start = time.time()

    # 1. Render cameras
    render_start = time.time()
    get_image_time = 0.0
    if args.single_camera:
        _, t = render_camera('third_person', args.renderer)
        get_image_time += t
    else:
        _, t = render_camera('third_person', args.renderer)
        get_image_time += t
        _, t = render_camera('top_down', args.renderer)
        get_image_time += t
        _, t = render_camera('wrist_cam', args.renderer)
        get_image_time += t
    render_time = time.time() - render_start

    # 2. Generate random actions (stand-in for VLA)
    robot_actions = np.random.randn(num_dof) * 0.01

    # 3. Apply actions and step physics
    physics_start = time.time()
    current_positions = []
    for joint_idx in controllable_joints[:num_dof]:
        current_positions.append(p.getJointState(robot_id, joint_idx)[0])

    for i, joint_idx in enumerate(controllable_joints[:num_dof]):
        target = current_positions[i] + robot_actions[i]
        target = np.clip(target, -np.pi, np.pi)
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=target, force=50)

    for _ in range(4):
        p.stepSimulation()
    physics_time = time.time() - physics_start

    total_time = time.time() - iter_start

    profile_data['render'].append(render_time)
    profile_data['get_image'].append(get_image_time)
    profile_data['physics'].append(physics_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{args.steps} ({total_time*1000:.1f} ms)")

    if GUI_MODE:
        time.sleep(0.01)

# Results
num_completed = len(profile_data['total'])
print(f"\n✓ Completed {num_completed} steps")

if num_completed > 0:
    print("\n" + "=" * 60)
    print("Performance Breakdown (average per iteration)")
    print("=" * 60)

    num_cams = 1 if args.single_camera else 3
    components = [
        (f'Rendering ({num_cams} camera{"" if num_cams == 1 else "s"})', 'render'),
        ('  getCameraImage()', 'get_image'),
        ('Physics step', 'physics'),
        ('Total iteration', 'total')
    ]

    total_avg = np.mean(profile_data['total']) * 1000

    for label, key in components:
        times = profile_data[key]
        avg = np.mean(times) * 1000
        min_t = np.min(times) * 1000
        max_t = np.max(times) * 1000
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"  {label:.<30} {avg:>7.2f} ms  ({pct:>5.1f}%)")
        if key == 'total':
            print(f"    {'(min/max)':.<28} {min_t:>7.2f} / {max_t:.2f} ms")

    print(f"\n  Effective rate: {1000/total_avg:.1f} Hz")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

try:
    p.disconnect()
except:
    pass
sys.exit(0)
