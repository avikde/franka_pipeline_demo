#!/usr/bin/env python3
"""
Fine-tune SmolVLA on custom SO-101 demonstrations.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Fine-tune SmolVLA')
parser.add_argument('--dataset-path', type=str, required=True,
                    help='Path to LeRobot dataset')
parser.add_argument('--output-dir', type=str, default='outputs/smolvla_finetuned',
                    help='Directory to save fine-tuned model')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--warmup-steps', type=int, default=100,
                    help='Warmup steps')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use')
args = parser.parse_args()

print("="*60)
print("SmolVLA Fine-Tuning")
print("="*60)
print(f"Dataset: {args.dataset_path}")
print(f"Output: {args.output_dir}")
print(f"Device: {args.device}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.lr}")
print("="*60 + "\n")

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(args.dataset_path)
print(f"Dataset size: {len(dataset)} frames")

# Load dataset statistics
stats_path = Path(args.dataset_path) / "dataset_stats.json"
if stats_path.exists():
    with open(stats_path) as f:
        dataset_stats = json.load(f)
    print(f"Loaded dataset statistics from {stats_path}")
else:
    print("Warning: No dataset_stats.json found. Using no normalization.")
    dataset_stats = None

# Split into train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Train size: {train_size}, Val size: {val_size}\n")

# Load pretrained model
print("Loading pretrained SmolVLA...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(args.device)

# Create preprocessor/postprocessor
preprocessor, postprocessor = make_smolvla_pre_post_processors(
    policy.config,
    dataset_stats=dataset_stats
)

print(f"Model loaded. Parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}\n")

# Freeze vision backbone (optional - can speed up training)
# Uncomment to freeze:
# for param in policy.model.vision_model.parameters():
#     param.requires_grad = False
# print("Vision backbone frozen")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, policy.parameters()),
    lr=args.lr,
    weight_decay=0.01
)

total_steps = len(train_dataset) // args.batch_size * args.epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps
)

# Preprocessing function
def preprocess_batch(batch):
    """Preprocess a batch for SmolVLA."""
    batch_size = len(batch['action'])

    # Convert images to tensors
    images_cam1 = torch.stack([
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        for img in batch['observation.images.camera1']
    ])
    images_cam2 = torch.stack([
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        for img in batch['observation.images.camera2']
    ])
    images_cam3 = torch.stack([
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        for img in batch['observation.images.camera3']
    ])

    # Convert states and actions
    states = torch.tensor(batch['observation.state'], dtype=torch.float32)
    actions = torch.tensor(batch['action'], dtype=torch.float32)

    # Create observation dict
    observation = {
        'observation.images.camera1': images_cam1.to(args.device),
        'observation.images.camera2': images_cam2.to(args.device),
        'observation.images.camera3': images_cam3.to(args.device),
        'observation.state': states.to(args.device),
        'task': 'Pick the red cube and place it on the blue cube',  # Task instruction
    }

    # Apply preprocessing
    processed_obs = preprocessor(observation)

    return processed_obs, actions.to(args.device)

# Training loop
print("Starting training...\n")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float('inf')

for epoch in range(args.epochs):
    # Training
    policy.train()
    train_losses = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]}
    )

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in pbar:
        try:
            obs, actions = preprocess_batch(batch)

            # Forward pass
            output = policy.forward(obs)
            pred_actions = output['action']

            # Compute loss (MSE between predicted and actual actions)
            loss = torch.nn.functional.mse_loss(pred_actions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"\nError in batch: {e}")
            continue

    avg_train_loss = np.mean(train_losses)

    # Validation
    policy.eval()
    val_losses = []

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]}
    )

    with torch.no_grad():
        for batch in val_loader:
            try:
                obs, actions = preprocess_batch(batch)
                output = policy.forward(obs)
                pred_actions = output['action']
                loss = torch.nn.functional.mse_loss(pred_actions, actions)
                val_losses.append(loss.item())
            except Exception as e:
                continue

    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')

    print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = output_dir / "best_model"
        policy.save_pretrained(checkpoint_path)
        print(f"✅ Saved best model to {checkpoint_path}\n")

# Save final model
final_path = output_dir / "final_model"
policy.save_pretrained(final_path)

print("\n" + "="*60)
print("Training Complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Models saved to: {args.output_dir}")
print("="*60)
print(f"\nTo use the fine-tuned model:")
print(f"  policy = SmolVLAPolicy.from_pretrained('{output_dir}/best_model')")
