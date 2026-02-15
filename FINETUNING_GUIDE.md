# SmolVLA Fine-Tuning Guide for SO-101

This guide shows you how to fine-tune SmolVLA using only your simulator and keyboard/mouse.

## Overview

**The Process:**
1. Collect 25-50 demonstrations using keyboard control ⌨️
2. Convert demos to LeRobot dataset format 📦
3. Fine-tune SmolVLA on your data 🔧
4. Use the fine-tuned model for inference 🚀

---

## Step 1: Collect Demonstrations

Run the teleoperation interface:

```bash
python scripts/teleop_so101.py --output-dir data/so101_demos --task "pick_red_cube_place_on_blue"
```

### Keyboard Controls:

⚠️ **IMPORTANT:** Click on the viewer window to give it focus!

**Joint Control:**
- `↑/↓` - Joint 0: Shoulder pan (rotate base)
- `←/→` - Joint 1: Shoulder lift (raise/lower arm)
- `1` + `↑/↓` - Joint 2: Elbow flex (bend elbow)
- `2` + `↑/↓` - Joint 3: Wrist flex (wrist up/down)
- `3` + `↑/↓` - Joint 4: Wrist roll (rotate wrist)
- `4` + `↑/↓` - Joint 5: Gripper (open/close)

**Recording:**
- `SPACE` - Start/stop recording episode
- `R` - Reset robot to starting position
- `ESC` - Quit and save

### Recording Workflow:

1. **Press `R`** to reset the robot
2. **Press `SPACE`** to start recording
3. **Complete the task** (e.g., pick red cube → place on blue cube)
4. **Press `SPACE`** to stop and save the episode
5. **Repeat 25-50 times** for good results

**Tips:**
- Each episode should be a complete task execution
- Try to be consistent but natural
- More diverse demonstrations = better generalization
- Aim for 25-50 successful episodes

---

## Step 2: Convert to LeRobot Dataset

Convert your raw demonstrations to LeRobot format:

```bash
python scripts/convert_to_lerobot_dataset.py \
  --input-dir data/so101_demos \
  --output-name so101_custom_task
```

This creates a dataset at `data/so101_custom_task/` with:
- Multi-view images (3 cameras)
- Robot states (joint positions)
- Actions (target positions)
- Dataset statistics for normalization

**Optional:** Push to HuggingFace Hub:
```bash
python scripts/convert_to_lerobot_dataset.py \
  --input-dir data/so101_demos \
  --output-name so101_custom_task \
  --push-to-hub \
  --repo-id YOUR_USERNAME/so101-custom-task
```

---

## Step 3: Fine-Tune SmolVLA

Fine-tune the pretrained model on your demonstrations:

```bash
python scripts/finetune_smolvla.py \
  --dataset-path data/so101_custom_task \
  --output-dir outputs/smolvla_finetuned \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-4
```

**Training Parameters:**
- `--epochs`: Number of passes through the data (default: 10)
- `--batch-size`: Samples per batch (default: 4, increase if you have GPU memory)
- `--lr`: Learning rate (default: 1e-4)
- `--warmup-steps`: Warmup steps for learning rate (default: 100)

**Expected Training Time:**
- With 50 episodes (~500 frames): ~10-20 minutes on GPU
- CPU training: Much slower (1-2 hours)

The best model will be saved to `outputs/smolvla_finetuned/best_model/`

---

## Step 4: Use Fine-Tuned Model

Update your demo script to use the fine-tuned model:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load fine-tuned model instead of base model
policy = SmolVLAPolicy.from_pretrained("outputs/smolvla_finetuned/best_model")
policy = policy.to(device).eval()

# Rest of inference code stays the same...
```

Or run the demo with your fine-tuned model:

```bash
python scripts/demo_smolvla_so101.py --steps 100
# (After modifying the script to load your fine-tuned model)
```

---

## Tips for Better Results

### Data Collection:
- ✅ **Be consistent** in your demonstrations
- ✅ **Vary starting conditions** slightly (different cube positions)
- ✅ **Complete the task** in each episode
- ✅ **Smooth movements** work better than jerky ones
- ❌ Don't include failed attempts (reset and re-record)

### Fine-Tuning:
- Start with **10 epochs**, increase if underfitting
- Monitor **validation loss** - should decrease over time
- If overfitting (val loss increases), reduce epochs or add regularization
- With limited data (< 25 demos), consider freezing the vision backbone

### Troubleshooting:
- **Robot doesn't move:** Check that actions are being applied correctly
- **Erratic behavior:** Collect more diverse demonstrations
- **Poor generalization:** Add more variation in demonstrations
- **Out of memory:** Reduce batch size

---

## Expected Results

With **25-50 demonstrations**:
- ✅ Should complete the specific task reliably
- ✅ May generalize to small variations
- ❌ Won't generalize to completely different tasks

With **100+ demonstrations**:
- ✅ Better generalization
- ✅ More robust to noise and variations

---

## Next Steps

After successful fine-tuning:

1. **Test extensively** in simulation
2. **Collect more data** for tasks where it fails
3. **Try different tasks** with new demonstrations
4. **Iterate:** Fine-tune → Test → Collect more data → Repeat

Good luck! 🚀
