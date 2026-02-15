#!/usr/bin/env python3
"""
SO-101 Teleoperation using MuJoCo's Native UI Controls

Uses the built-in MuJoCo viewer control panel (right side).
Much simpler and more reliable than custom keyboard control!

Instructions:
1. Open the right panel (Ctrl+Right arrow or click the arrow on right edge)
2. Go to "Control" section
3. Use sliders to control joints
4. Press SPACE to start/stop recording
5. Press R to reset robot
"""

import argparse
import mujoco
import numpy as np
import time
import json
from pathlib import Path
from PIL import Image
import sys

# Parse arguments
parser = argparse.ArgumentParser(description='SO-101 Teleoperation (MuJoCo UI)')
parser.add_argument('--output-dir', type=str, default='data/so101_demos',
                    help='Directory to save demonstrations')
parser.add_argument('--task', type=str, default='pick_red_cube_place_on_blue',
                    help='Task name for the dataset')
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("🤖 SO-101 Teleoperation (MuJoCo Native UI)")
print("="*70)

# Load robot model (with tuned gains and contact physics)
print("Loading robot model...")
model = mujoco.MjModel.from_xml_path('assets/so101/so101_vision_scene.xml')
data = mujoco.MjData(model)

# Rendering setup for data collection
IMG_WIDTH, IMG_HEIGHT = 256, 256
model.vis.global_.offwidth = max(model.vis.global_.offwidth, 512)
model.vis.global_.offheight = max(model.vis.global_.offheight, 512)
renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

def render_camera(camera_name):
    """Render from a specific camera."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

# State variables
recording = False
episode_data = []
current_episode = 0
frame_index = 0
last_space_time = 0
last_r_time = 0

def reset_robot():
    """Reset robot to initial position."""
    global frame_index
    mujoco.mj_resetData(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
    frame_index = 0
    print("✓ Robot reset")

def save_episode():
    """Save the current episode."""
    global episode_data, current_episode

    if len(episode_data) == 0:
        print("! No data to save")
        return

    episode_path = output_dir / f"episode_{current_episode:04d}"
    episode_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving episode {current_episode}...")

    # Save images
    for i, frame in enumerate(episode_data):
        for cam_name in ['third_person', 'top_down', 'wrist_cam']:
            img = Image.fromarray(frame['images'][cam_name])
            img.save(episode_path / f"frame_{i:04d}_{cam_name}.png")

    # Save states and actions
    states = np.array([frame['state'] for frame in episode_data])
    actions = np.array([frame['action'] for frame in episode_data])
    np.save(episode_path / "states.npy", states)
    np.save(episode_path / "actions.npy", actions)

    # Save metadata
    metadata = {
        'episode_index': current_episode,
        'num_frames': len(episode_data),
        'task': args.task,
        'fps': 10,
    }
    with open(episode_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved episode {current_episode} ({len(episode_data)} frames)")
    current_episode += 1
    episode_data = []

# Key callback for recording control
def key_callback(keycode):
    """Handle keyboard input for recording control."""
    global recording, last_space_time, last_r_time

    current_time = time.time()

    # Space - toggle recording (with debounce)
    if keycode == 32:  # SPACE
        if current_time - last_space_time > 0.5:
            last_space_time = current_time
            recording = not recording
            if recording:
                print("\n" + "="*70)
                print("🔴 RECORDING STARTED")
                print("="*70)
                episode_data.clear()
                frame_index = 0
            else:
                print("\n" + "="*70)
                print("⏹️  RECORDING STOPPED")
                print("="*70)
                if len(episode_data) > 0:
                    save_episode()

    # R - reset robot (with debounce)
    elif keycode == 82 or keycode == 114:  # R or r
        if current_time - last_r_time > 0.5:
            last_r_time = current_time
            reset_robot()

# Initialize
reset_robot()

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Open the RIGHT PANEL in the viewer:")
print("   - Press Ctrl+Right arrow")
print("   - Or click the arrow (→) on the right edge of the window")
print()
print("2. In the right panel, find the 'Control' section")
print()
print("3. Use the sliders to control the 6 robot joints:")
print("   - shoulder_pan")
print("   - shoulder_lift")
print("   - elbow_flex")
print("   - wrist_flex")
print("   - wrist_roll")
print("   - gripper")
print()
print("4. KEYBOARD CONTROLS:")
print("   - SPACE: Start/stop recording episode")
print("   - R: Reset robot to initial position")
print("   - ESC: Quit")
print("="*70)
print("\n⏳ Launching viewer...")

# Launch viewer with UI enabled
try:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,      # Hide left UI (clean)
        show_right_ui=True,      # SHOW right UI (has controls!)
        key_callback=key_callback
    )
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 35
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0.2, 0.0, 0.2]

    print("✅ Viewer launched!")
    print("\n💡 TIP: Open the right panel now (Ctrl+Right arrow)")
    print("="*70 + "\n")

except Exception as e:
    print(f"❌ Error launching viewer: {e}")
    sys.exit(1)

# Main simulation loop
last_record_time = time.time()

try:
    while viewer.is_running():
        # Step simulation (viewer controls will update data.ctrl)
        mujoco.mj_step(model, data)

        # Record data at ~10 Hz
        current_time = time.time()
        if recording and (current_time - last_record_time) >= 0.1:
            last_record_time = current_time

            # Capture images
            images = {
                'third_person': render_camera('third_person'),
                'top_down': render_camera('top_down'),
                'wrist_cam': render_camera('wrist_cam'),
            }

            # Record state and action (ctrl is set by viewer sliders)
            frame_data = {
                'state': data.qpos[:6].copy(),
                'action': data.ctrl[:6].copy(),  # Current control values from sliders
                'images': images,
                'timestamp': current_time,
            }
            episode_data.append(frame_data)

        # Sync viewer
        viewer.sync()
        time.sleep(0.002)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted!")

# Save any unsaved data
if recording and len(episode_data) > 0:
    print("\n💾 Saving current episode...")
    save_episode()

# Cleanup
viewer.close()

print("\n" + "="*70)
print(f"✅ Session complete!")
print(f"   Episodes saved: {current_episode}")
print(f"   Output directory: {output_dir}")
print("="*70)
print("\nNext steps:")
print(f"  python scripts/convert_to_lerobot_dataset.py --input-dir {output_dir}")
print("\nGoodbye! 👋")
