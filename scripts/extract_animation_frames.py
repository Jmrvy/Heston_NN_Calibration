#!/usr/bin/env python3
"""
Extract frames from nn_training_animation.gif for LaTeX animate package.
Run this script to generate the frames needed for the presentation.
"""

from pathlib import Path
import os

try:
    from PIL import Image
except ImportError:
    print("Error: PIL/Pillow not found. Install with: pip install Pillow")
    exit(1)

# Determine project root (where this script is located)
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Paths relative to project root
gif_path = project_root / 'master_dissertation_ppt' / 'figures' / 'nn_training_animation.gif'
frames_dir = project_root / 'master_dissertation_ppt' / 'figures' / 'nn_training_frames'

if not gif_path.exists():
    print(f"Error: GIF not found at {gif_path}")
    exit(1)

# Create frames directory
frames_dir.mkdir(exist_ok=True, parents=True)

print(f"Extracting frames from {gif_path.relative_to(project_root)}...")
gif_img = Image.open(gif_path)
frame_count = 0

try:
    while True:
        # Save frame with zero-padded number (000, 001, 002, ...)
        frame_path = frames_dir / f"nn_training_animation_{frame_count:03d}.png"
        gif_img.save(frame_path)
        frame_count += 1
        gif_img.seek(frame_count)
except (EOFError, ValueError):
    pass

print(f" Extracted {frame_count} frames to {frames_dir.relative_to(project_root)}/")
print(f"   First frame: nn_training_animation_000.png")
print(f"   Last frame: nn_training_animation_{frame_count-1:03d}.png")
print(f"\nNow update slide.tex to use:")
print(f"   \\animategraphics[loop,autoplay,width=0.85\\textwidth]{{20}}{{figures/nn_training_frames/nn_training_animation_}}{{000}}{{{frame_count-1}}}")