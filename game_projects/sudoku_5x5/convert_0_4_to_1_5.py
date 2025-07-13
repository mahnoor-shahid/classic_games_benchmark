import os
import json
import argparse
import numpy as np
from typing import List, Dict
from PIL import Image

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torchvision = None

def load_puzzles(json_path: str) -> Dict[str, List[Dict]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Handle both cases: list of puzzles or dict with "0_4"/"5_9" keys
    if isinstance(data, list):
        return {"0_4": data}  # Assume it's 0_4 if it's a plain list
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError("JSON must be a list of puzzles or a dict with puzzle sets")

def shift_grid(grid, shift=1):
    # Add 1 to all non-empty cells (-1 stays -1)
    return [[cell + shift if cell != -1 else -1 for cell in row] for row in grid]

def save_puzzles(puzzles: List[Dict], out_path: str):
    with open(out_path, 'w') as f:
        json.dump(puzzles, f, indent=2)

def load_mnist_images_1_5(mnist_root: str) -> Dict[int, np.ndarray]:
    if torchvision is None:
        raise ImportError("torchvision is required for MNIST image regeneration.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
    ])
    train_dataset = torchvision.datasets.MNIST(
        root=mnist_root, train=True, download=True, transform=transform
    )
    digit_images = {}
    for image, label in train_dataset:
        if 1 <= label <= 5 and label not in digit_images:
            digit_images[label] = image
            if len(digit_images) == 5:
                break
    # Fallback for missing digits
    for d in range(1, 6):
        if d not in digit_images:
            digit_images[d] = np.zeros((28, 28), dtype=np.uint8)
    return digit_images

def make_mnist_grid(grid, mnist_images):
    img = np.zeros((140, 140), dtype=np.uint8)
    for row in range(5):
        for col in range(5):
            digit = grid[row][col]
            if digit == -1:
                continue
            mnist_img = mnist_images.get(digit, np.zeros((28, 28), dtype=np.uint8))
            img[row*28:(row+1)*28, col*28:(col+1)*28] = mnist_img
    return img

def process_puzzle_set(puzzles: List[Dict], output_base: str, images_dir: str = None, mnist_images: Dict = None):
    """Process a single set of puzzles (0_4 or 5_9)"""
    new_puzzles = []
    for entry in puzzles:
        if isinstance(entry, list):
            # Entry is a puzzle grid (list of lists)
            new_grid = shift_grid(entry, shift=1)
            new_puzzles.append(new_grid)
        elif isinstance(entry, dict):
            # Entry is a dictionary with metadata
            new_entry = entry.copy()
            # Shift puzzle and solution grids
            if 'puzzle_grid' in entry:
                new_entry['puzzle_grid'] = shift_grid(entry['puzzle_grid'], shift=1)
            if 'solution_grid' in entry:
                new_entry['solution_grid'] = shift_grid(entry['solution_grid'], shift=1)
            new_puzzles.append(new_entry)
        else:
            print(f"⚠️ Skipping unsupported entry type: {type(entry)}")
            continue
    
    # Save shifted puzzles
    save_puzzles(new_puzzles, output_base)
    print(f"✅ Saved shifted puzzles to {output_base}")
    
    # Generate images if requested
    if images_dir and mnist_images:
        os.makedirs(images_dir, exist_ok=True)
        for i, entry in enumerate(new_puzzles):
            if isinstance(entry, list):
                # Entry is a puzzle grid
                puzzle_img = make_mnist_grid(entry, mnist_images)
                Image.fromarray(puzzle_img).save(os.path.join(images_dir, f'puzzle_{i:04d}.png'))
            elif isinstance(entry, dict):
                # Entry is a dictionary with metadata
                pid = entry.get('id', f'puzzle_{i:04d}')
                if 'puzzle_grid' in entry:
                    puzzle_img = make_mnist_grid(entry['puzzle_grid'], mnist_images)
                    Image.fromarray(puzzle_img).save(os.path.join(images_dir, f'{pid}_puzzle.png'))
                if 'solution_grid' in entry:
                    solution_img = make_mnist_grid(entry['solution_grid'], mnist_images)
                    Image.fromarray(solution_img).save(os.path.join(images_dir, f'{pid}_solution.png'))
        print(f"✅ Regenerated MNIST images saved to {images_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert 5x5 Sudoku puzzles from digits 0-4 to 1-5.")
    parser.add_argument('--input', required=True, help='Input JSON file with puzzles (digits 0-4)')
    parser.add_argument('--output', required=True, help='Output JSON file for puzzles (digits 1-5)')
    parser.add_argument('--regen-images', action='store_true', help='Regenerate MNIST images for digits 1-5')
    parser.add_argument('--mnist-root', default='../../shared_data', help='Root directory for MNIST data')
    parser.add_argument('--images-dir', default=None, help='Directory to save regenerated images (if --regen-images)')
    args = parser.parse_args()

    # Load puzzles
    puzzle_sets = load_puzzles(args.input)
    print(f"📁 Found puzzle sets: {list(puzzle_sets.keys())}")
    
    # Load MNIST images if needed
    mnist_images = None
    if args.regen_images:
        if torchvision is None:
            print("torchvision is not installed. Cannot regenerate images.")
            return
        mnist_images = load_mnist_images_1_5(args.mnist_root)
    
    # Process each set
    for set_name, puzzles in puzzle_sets.items():
        print(f"\n🔄 Processing {set_name} set ({len(puzzles)} puzzles)...")
        
        # Determine output paths
        if len(puzzle_sets) == 1:
            # Single set, use the provided output path
            output_path = args.output
            images_path = args.images_dir
        else:
            # Multiple sets, create separate files
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1]
            output_path = f"{base_name}_{set_name}{ext}"
            images_path = f"{args.images_dir}_{set_name}/" if args.images_dir else None
        
        process_puzzle_set(puzzles, output_path, images_path, mnist_images)

if __name__ == '__main__':
    main() 