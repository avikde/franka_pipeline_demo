import jax

print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())

import mujoco
from mujoco import mjx
import time

print("Loading simplified Franka model...")
model = mujoco.MjModel.from_xml_path('../franka_mjx_simple.xml')

print("Transferring to GPU...")
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

print("Compiling (expecting 10-30 seconds)...")
step_jit = jax.jit(lambda d: mjx.step(mjx_model, d))

start = time.time()
mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()
compile_time = time.time() - start

print(f"✓ Compiled in {compile_time:.1f}s")

# Test speed
start = time.time()
for _ in range(1000):
    mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()
elapsed = time.time() - start

print(f"✓ 1000 steps in {elapsed:.3f}s ({1000/elapsed:.0f} steps/sec)")
print(f"Joint positions: {mjx_data.qpos[:7]}")