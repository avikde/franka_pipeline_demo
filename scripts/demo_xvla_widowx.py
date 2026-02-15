#!/usr/bin/env python3
"""
X-VLA with WidowX Robot Demo

Uses the X-VLA WidowX checkpoint (lerobot/xvla-widowx) with the WidowX robot model.
This checkpoint is specifically fine-tuned on BridgeData for WidowX pick-and-place tasks.

WidowX is a 6-DoF robot (5 arm joints + gripper), matching the checkpoint's training data.
"""

import argparse
import mujoco
import numpy as np
import torch
from PIL import Image
import time
import sys

# Parse arguments
parser = argparse.ArgumentParser(description='X-VLA with WidowX Robot')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
parser.add_argument('--headless', action='store_true', help='Run without GUI')
args = parser.parse_args()

print("X-VLA WidowX Demo")

# Load X-VLA policy
print("\nLoading X-VLA WidowX policy...")
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load WidowX-specific checkpoint
    policy = XVLAPolicy.from_pretrained("lerobot/xvla-widowx").to(device).eval()

    # Load language tokenizer (X-VLA uses BART)
    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)

    print(f"Loaded X-VLA (chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps})")

except ImportError as e:
    print(f"\n❌ X-VLA not installed: {e}")
    print("\nTo install X-VLA:")
    print('  pip install "lerobot[xvla]"')
    print("\nAlternatively, run SmolVLA demo:")
    print("  python scripts/demo_vla_inference.py")
    sys.exit(1)

# Load WidowX MuJoCo model
try:
    model = mujoco.MjModel.from_xml_path('assets/widowx/widowx_vision_scene.xml')
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading WidowX model: {e}")
    sys.exit(1)

# Setup renderer
VLA_WIDTH, VLA_HEIGHT = 256, 256
model.vis.global_.offwidth = max(model.vis.global_.offwidth, VLA_WIDTH)
model.vis.global_.offheight = max(model.vis.global_.offheight, VLA_HEIGHT)
renderer = mujoco.Renderer(model, height=VLA_HEIGHT, width=VLA_WIDTH)

def render_camera(camera_name):
    """Render from a specific camera."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def preprocess_image(rgb_image, device='cpu'):
    """Preprocess image for VLA input."""
    img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

# Task instruction
task_instruction = "Pick up the red block"
print(f"Task: '{task_instruction}'")

# Tokenize language instruction once (before the loop)
tokenized = tokenizer(
    task_instruction,
    padding='max_length',
    max_length=policy.config.tokenizer_max_length,
    truncation=True,
    return_tensors='pt'
)
language_tokens = tokenized['input_ids'].to(device)
language_attention_mask = tokenized['attention_mask'].to(device)

# Settle physics
for _ in range(100):
    mujoco.mj_step(model, data)

# Launch viewer if GUI mode
viewer = None
if not args.headless:
    try:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.2, 0.0, 0.2]
    except Exception as e:
        args.headless = True

# Simulation loop
print("Running X-VLA inference loop...")

# Profiling
profile_data = {
    'render': [],
    'vla_inference': [],
    'physics': [],
    'total': []
}

for step in range(args.steps):
    iter_start = time.time()

    # 1. Render cameras (matching X-VLA WidowX training: "up" and "side")
    render_start = time.time()
    try:
        img_up = render_camera('up')
        img_side = render_camera('side')
    except:
        # Fallback if cameras not found
        img_up = render_camera('third_person')
        img_side = img_up
    render_time = time.time() - render_start

    # 2. Preprocess for X-VLA
    img_up_tensor = preprocess_image(img_up, device=device)
    img_side_tensor = preprocess_image(img_side, device=device)

    # Get robot state (6 arm joint positions - excluding gripper for state)
    # WX250S has joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
    robot_qpos = data.qpos[:6]  # First 6 joints (arm only)

    # X-VLA WidowX expects specific observation keys
    # From policy.config: 'observation.images.image' and 'observation.images.image2'
    observation = {
        'observation.images.image': img_up_tensor,       # Camera 1
        'observation.images.image2': img_side_tensor,    # Camera 2
        'observation.state': torch.from_numpy(robot_qpos).float().unsqueeze(0).to(device),
        'observation.language.tokens': language_tokens,
        'observation.language.attention_mask': language_attention_mask,
    }

    # 3. VLA inference (check if new chunk or popping from queue)
    vla_start = time.time()
    queue_was_empty = len(policy._queues.get("action", [])) == 0

    with torch.inference_mode():
        try:
            actions = policy.select_action(observation)
        except Exception as e:
            print(f"VLA inference error at step {step}: {e}")
            actions = torch.zeros(7, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    vla_time = time.time() - vla_start

    # Convert actions to numpy
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().flatten()
    else:
        actions_np = np.array(actions).flatten()

    # Extract robot actions (7-DoF: 6 arm joints + gripper)
    # X-VLA outputs actions for the robot's DoF
    robot_actions = actions_np[:7] if len(actions_np) >= 7 else np.pad(actions_np, (0, 7-len(actions_np)))

    # Apply actions to WidowX WX250S (position control)
    # Scale actions appropriately for the robot's range
    data.ctrl[:model.nu] = np.clip(robot_actions[:model.nu] * 0.1, -1.0, 1.0)

    # 4. Step simulation
    physics_start = time.time()
    mujoco.mj_step(model, data)
    physics_time = time.time() - physics_start

    # 5. Viewer sync
    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            print(f"\nViewer closed at step {step}")
            break

    # Total iteration time
    total_time = time.time() - iter_start

    # Store profiling data
    profile_data['render'].append(render_time)
    profile_data['vla_inference'].append(vla_time)
    profile_data['physics'].append(physics_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{args.steps}: actions = [{robot_actions[0]:.3f}, {robot_actions[1]:.3f}, {robot_actions[2]:.3f}] ({total_time*1000:.1f} ms)")

    if total_time > 0.2:
        print(f"  !! SLOW step {step} ({total_time*1000:.1f} ms) new_chunk={queue_was_empty} — "
              f"render={render_time*1000:.1f}ms, vla={vla_time*1000:.1f}ms, physics={physics_time*1000:.1f}ms")

# Cleanup
if viewer is not None:
    viewer.close()

# Print timing statistics
num_completed = len(profile_data['total'])
print(f"\nCompleted {num_completed} simulation steps")

if num_completed > 0:
    print("\nPerformance Breakdown (average per iteration):")
    print("-" * 50)

    components = [
        ('Rendering (2 cameras)', 'render'),
        ('VLA inference', 'vla_inference'),
        ('Physics step', 'physics'),
        ('Total iteration', 'total')
    ]

    total_avg = np.mean(profile_data['total']) * 1000

    for label, key in components:
        times = profile_data[key]
        avg = np.mean(times) * 1000
        percentage = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"  {label:.<30} {avg:>7.2f} ms  ({percentage:>5.1f}%)")

    print(f"\n  Effective rate: {1000/total_avg:.1f} Hz" if total_avg > 0 else "")

print("\nDemo complete!")
