import jax

print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())

import mujoco
from mujoco import mjx
import time

import jax
import mujoco
from mujoco import mjx

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
mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()

print("MJX working!")
print(f"Position: {mjx_data.qpos}")

print("Loading model...")
# Use the simpler panda.xml instead of scene.xml
model = mujoco.MjModel.from_xml_path('../mujoco_menagerie/franka_emika_panda/panda.xml')

print("Transferring to GPU...")
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

print("Compiling step function (this may take 30-60 seconds on first run)...")
def step_fn(data):
    return mjx.step(mjx_model, data)

# JIT compile
step_jit = jax.jit(step_fn)

# First call triggers compilation - will be slow
print("Running first step (compiling)...")
start = time.time()
mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()  # Wait for GPU to finish
print(f"First step took {time.time() - start:.2f}s (includes compilation)")

# Second call should be fast
print("Running second step (compiled)...")
start = time.time()
mjx_data = step_jit(mjx_data)
mjx_data.qpos.block_until_ready()
print(f"Second step took {time.time() - start:.4f}s")

print("\nMJX GPU setup working!")
print(f"Joint positions: {mjx_data.qpos}")
