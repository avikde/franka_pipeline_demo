import mujoco
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('../franka_mjx_simple.xml')
data = mujoco.MjData(model)

# Create offscreen renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation loop
for step in range(100):
    mujoco.mj_step(model, data)
    
    # Render every 10 steps
    if step % 10 == 0:
        renderer.update_scene(data)
        pixels = renderer.render()
        
        # Save or display
        plt.imsave(f'frame_{step:04d}.png', pixels)

print("Rendered 10 frames to frame_*.png")