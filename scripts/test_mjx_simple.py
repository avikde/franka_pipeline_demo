import jax

print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())

import mujoco
from mujoco import mjx
import time

# Use a built-in simple model instead
xml = """
<mujoco>
  <worldbody>
    <body>
      <joint type="hinge"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

print("Testing MJX with simple pendulum...")
model = mujoco.MjModel.from_xml_string(xml)
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

# Run one step
step_jit = jax.jit(lambda d: mjx.step(mjx_model, d))
# mjx_data = step_jit(mjx_data)
# mjx_data.qpos.block_until_ready()

# First call triggers compilation - will be slow
print("Running first step (compiling)...")
start = time.time()
mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()  # Wait for GPU to finish
print(f"First step took {time.time() - start:.2f}s (includes compilation)")

# Second call should be fast
print("Running second step (compiled)...")
start = time.time()
for iter_cnt in range(1000):
    mjx_data = step_jit(mjx_data)
    mjx_data.qpos.block_until_ready()
print(f"Avg took {time.time() - start:.4f}ms")

print("MJX working!")
print(f"Position: {mjx_data.qpos}")
