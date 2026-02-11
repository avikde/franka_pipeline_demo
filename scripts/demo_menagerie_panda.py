import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load model with the proper Franka Emika Panda (MJX version from mujoco_menagerie)
model = mujoco.MjModel.from_xml_path('assets/panda_vision_scene.xml')
data = mujoco.MjData(model)

# Create renderer
WIDTH, HEIGHT = 640, 480
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

def render_camera(camera_name):
    """Render from a specific camera."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    pixels = renderer.render()
    return pixels

def detect_red_cube_simple(rgb_image):
    """
    Simple color-based detection for the red cube.
    Returns: (x, y, confidence) in pixel coordinates, or None if not found.
    """
    # Convert to float and normalize
    img = rgb_image.astype(np.float32) / 255.0

    # Red mask: high R, low G and B
    red_mask = (img[:, :, 0] > 0.6) & (img[:, :, 1] < 0.3) & (img[:, :, 2] < 0.3)

    # Find center of mass of red pixels
    if red_mask.sum() > 100:  # At least 100 red pixels
        y_coords, x_coords = np.where(red_mask)
        center_x = x_coords.mean()
        center_y = y_coords.mean()
        confidence = red_mask.sum() / (HEIGHT * WIDTH)
        return center_x, center_y, confidence

    return None

def pixel_to_3d_position(pixel_x, pixel_y, camera_name):
    """
    Convert pixel coordinates to 3D world position.
    This is a simplified version - in practice you'd use depth info.
    For now, we'll use the known cube height (z=0.05) and ray casting.
    """
    camera_id = model.camera(camera_name).id

    # Get camera pose
    cam_pos = np.zeros(3)
    cam_mat = np.zeros(9)
    mujoco.mj_forward(model, data)
    mujoco.mjv_cameraInModel(cam_pos, cam_mat, model, data, camera_id)

    # Simple approximation: assume cube is at table height (z=0.05)
    # and use pinhole camera model
    # For a proper solution, you'd use depth rendering or known object sizes

    # This is a placeholder - we'll refine this with proper depth or VLM
    return np.array([0.5, 0.0, 0.05])  # Return approximate cube position

# Simulation and visualization
print("Running simulation with vision pipeline using mujoco_menagerie Franka Panda...")

# Run a few steps to settle physics
for _ in range(100):
    mujoco.mj_step(model, data)

# Render from different cameras
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Third person view
img_third = render_camera('third_person')
axes[0, 0].imshow(img_third)
axes[0, 0].set_title('Third Person View')
axes[0, 0].axis('off')

# Top down view
img_top = render_camera('top_down')
axes[0, 1].imshow(img_top)
axes[0, 1].set_title('Top Down View')
axes[0, 1].axis('off')

# Wrist camera view
img_wrist = render_camera('wrist_cam')
axes[1, 0].imshow(img_wrist)
axes[1, 0].set_title('Wrist Camera View')
axes[1, 0].axis('off')

# Red cube detection on third person view
detection = detect_red_cube_simple(img_third)
if detection:
    px, py, conf = detection
    axes[1, 1].imshow(img_third)
    axes[1, 1].plot(px, py, 'g+', markersize=20, markeredgewidth=3)
    axes[1, 1].set_title(f'Red Cube Detected (conf: {conf:.3f})')
    print(f"Red cube detected at pixel ({px:.1f}, {py:.1f}), confidence: {conf:.3f}")
else:
    axes[1, 1].imshow(img_third)
    axes[1, 1].set_title('Red Cube NOT Detected')
    print("Red cube not detected")
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('menagerie_panda_camera_views.png', dpi=150)
print("Saved camera views to menagerie_panda_camera_views.png")

# Print cube position from simulation (ground truth)
red_cube_id = model.body('red_cube').id
cube_pos = data.xpos[red_cube_id]
print(f"Ground truth red cube position: {cube_pos}")

# Print robot info
print(f"\nRobot info:")
print(f"  Total bodies: {model.nbody}")
print(f"  Total joints: {model.njnt}")
print(f"  Total actuators: {model.nu}")
