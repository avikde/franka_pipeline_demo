#!/usr/bin/env python3
"""
Convert teleoperation demonstrations to LeRobot dataset format.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image as ImageFeature
import torch

parser = argparse.ArgumentParser(description='Convert demos to LeRobot dataset')
parser.add_argument('--input-dir', type=str, default='data/so101_demos',
                    help='Directory with raw demonstrations')
parser.add_argument('--output-name', type=str, default='so101_custom_task',
                    help='Name for the dataset')
parser.add_argument('--push-to-hub', action='store_true',
                    help='Push to HuggingFace Hub')
parser.add_argument('--repo-id', type=str, default=None,
                    help='HuggingFace repo ID (e.g., username/dataset-name)')
args = parser.parse_args()

input_dir = Path(args.input_dir)
if not input_dir.exists():
    print(f"Error: Input directory {input_dir} does not exist!")
    exit(1)

# Find all episodes
episode_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])

if len(episode_dirs) == 0:
    print(f"Error: No episodes found in {input_dir}")
    exit(1)

print(f"Found {len(episode_dirs)} episodes")

# Collect all data
all_data = []

for ep_idx, episode_dir in enumerate(episode_dirs):
    print(f"Processing {episode_dir.name}...")

    # Load metadata
    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_frames = metadata['num_frames']

    # Load states and actions
    states = np.load(episode_dir / "states.npy")
    actions = np.load(episode_dir / "actions.npy")

    # Load images for each frame
    for frame_idx in range(num_frames):
        # Load camera images
        img_third = Image.open(episode_dir / f"frame_{frame_idx:04d}_third_person.png")
        img_top = Image.open(episode_dir / f"frame_{frame_idx:04d}_top_down.png")
        img_wrist = Image.open(episode_dir / f"frame_{frame_idx:04d}_wrist_cam.png")

        # Create data entry in LeRobot format
        data_entry = {
            'observation.images.camera1': img_third,  # third_person
            'observation.images.camera2': img_top,     # top_down
            'observation.images.camera3': img_wrist,   # wrist_cam
            'observation.state': states[frame_idx].tolist(),
            'action': actions[frame_idx].tolist(),
            'episode_index': ep_idx,
            'frame_index': frame_idx,
            'timestamp': frame_idx / metadata.get('fps', 30.0),
            'next.done': frame_idx == num_frames - 1,
            'index': len(all_data),
            'task_index': 0,  # Single task for now
        }
        all_data.append(data_entry)

print(f"\nTotal frames collected: {len(all_data)}")
print(f"Total episodes: {len(episode_dirs)}")

# Define dataset features
features = Features({
    'observation.images.camera1': ImageFeature(),
    'observation.images.camera2': ImageFeature(),
    'observation.images.camera3': ImageFeature(),
    'observation.state': Sequence(Value('float32')),
    'action': Sequence(Value('float32')),
    'episode_index': Value('int64'),
    'frame_index': Value('int64'),
    'timestamp': Value('float32'),
    'next.done': Value('bool'),
    'index': Value('int64'),
    'task_index': Value('int64'),
})

# Create dataset
dataset = Dataset.from_list(all_data, features=features)

# Compute dataset statistics (needed for normalization)
states = np.array([item['observation.state'] for item in all_data])
actions = np.array([item['action'] for item in all_data])

stats = {
    'observation.state': {
        'mean': states.mean(axis=0).tolist(),
        'std': states.std(axis=0).tolist(),
        'min': states.min(axis=0).tolist(),
        'max': states.max(axis=0).tolist(),
    },
    'action': {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
    }
}

print("\nDataset Statistics:")
print(f"  State mean: {stats['observation.state']['mean']}")
print(f"  State std:  {stats['observation.state']['std']}")
print(f"  Action mean: {stats['action']['mean']}")
print(f"  Action std:  {stats['action']['std']}")

# Save dataset locally
output_path = input_dir.parent / args.output_name
dataset.save_to_disk(str(output_path))
print(f"\nDataset saved to: {output_path}")

# Save statistics
stats_path = output_path / "dataset_stats.json"
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Statistics saved to: {stats_path}")

# Push to hub if requested
if args.push_to_hub:
    if args.repo_id is None:
        print("Error: --repo-id required when using --push-to-hub")
        exit(1)

    print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
    dataset.push_to_hub(args.repo_id)
    print("Dataset pushed successfully!")

print("\n✅ Dataset creation complete!")
print(f"\nNext steps:")
print(f"1. Review dataset at: {output_path}")
print(f"2. Fine-tune SmolVLA with:")
print(f"   python scripts/finetune_smolvla.py --dataset-path {output_path}")
