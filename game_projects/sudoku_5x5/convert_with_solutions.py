import os
import json
import argparse
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import time

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torchvision = None

class SudokuSolver:
    """Simple 5x5 Sudoku solver"""
    
    def solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """Solve a 5x5 Sudoku puzzle"""
        grid = [row[:] for row in puzzle]  # Copy the puzzle
        
        def is_valid(grid, row, col, num):
            # Check row
            if num in grid[row]:
                return False
            # Check column
            if num in [grid[i][col] for i in range(5)]:
                return False
            # Check main diagonal
            if row == col and num in [grid[i][i] for i in range(5) if i != row]:
                return False
            # Check anti-diagonal
            if row + col == 4 and num in [grid[i][4-i] for i in range(5) if i != row]:
                return False
            return True
        
        def solve_grid(grid):
            for row in range(5):
                for col in range(5):
                    if grid[row][col] == -1:  # Empty cell
                        for num in range(5):
                            if is_valid(grid, row, col, num):
                                grid[row][col] = num
                                if solve_grid(grid):
                                    return True
                                grid[row][col] = -1
                        return False
            return True
        
        if solve_grid(grid):
            return grid
        return None

def shift_grid(grid, shift=1):
    """Add shift to all non-empty cells (-1 stays -1)"""
    return [[cell + shift if cell != -1 else -1 for cell in row] for row in grid]

def load_mnist_images_1_5(mnist_root: str) -> Dict[int, np.ndarray]:
    """Load MNIST images for digits 1-5"""
    if torchvision is None:
        raise ImportError("torchvision is required for MNIST image generation.")
    
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
    """Create a 140x140 MNIST image grid"""
    img = np.zeros((140, 140), dtype=np.uint8)
    for row in range(5):
        for col in range(5):
            digit = grid[row][col]
            if digit == -1:
                continue
            mnist_img = mnist_images.get(digit, np.zeros((28, 28), dtype=np.uint8))
            img[row*28:(row+1)*28, col*28:(col+1)*28] = mnist_img
    return img

def create_metadata(puzzle_grid, solution_grid, puzzle_id, difficulty="unvalidated"):
    """Create metadata for a puzzle"""
    filled_cells = sum(1 for row in puzzle_grid for cell in row if cell != -1)
    total_cells = 25
    
    return {
        "id": puzzle_id,
        "difficulty": difficulty,
        "grid_size": 5,
        "filled_cells": filled_cells,
        "empty_cells": total_cells - filled_cells,
        "completion_percentage": (filled_cells / total_cells) * 100,
        "generation_timestamp": time.time(),
        "digit_set": "1-5",
        "original_digit_set": "0-4"
    }

def main():
    parser = argparse.ArgumentParser(description="Convert 5x5 Sudoku puzzles from 0-4 to 1-5 with solutions and metadata.")
    parser.add_argument('--input', required=True, help='Input JSON file with puzzle grids (digits 0-4)')
    parser.add_argument('--output', required=True, help='Output JSON file for puzzles with solutions and metadata (digits 1-5)')
    parser.add_argument('--regen-images', action='store_true', help='Generate MNIST images for puzzles and solutions')
    parser.add_argument('--mnist-root', default='../../shared_data', help='Root directory for MNIST data')
    parser.add_argument('--images-dir', default=None, help='Directory to save generated images')
    parser.add_argument('--difficulty', default='unvalidated', help='Difficulty level for metadata')
    args = parser.parse_args()

    # Load puzzle grids
    with open(args.input, 'r') as f:
        puzzle_grids = json.load(f)
    
    print(f"📁 Loaded {len(puzzle_grids)} puzzle grids")
    
    # Initialize solver
    solver = SudokuSolver()
    
    # Process each puzzle
    processed_puzzles = []
    solved_count = 0
    failed_count = 0
    
    for i, puzzle_grid in enumerate(puzzle_grids):
        print(f"🔄 Processing puzzle {i+1}/{len(puzzle_grids)}...")
        
        # Solve the puzzle
        solution_grid = solver.solve(puzzle_grid)
        
        if solution_grid is None:
            print(f"❌ Failed to solve puzzle {i+1}")
            failed_count += 1
            continue
        
        solved_count += 1
        
        # Shift both puzzle and solution to 1-5
        shifted_puzzle = shift_grid(puzzle_grid, shift=1)
        shifted_solution = shift_grid(solution_grid, shift=1)
        
        # Create metadata
        puzzle_id = f"puzzle_{i+1:04d}"
        metadata = create_metadata(shifted_puzzle, shifted_solution, puzzle_id, args.difficulty)
        
        # Create complete puzzle entry
        puzzle_entry = {
            "id": puzzle_id,
            "puzzle_grid": shifted_puzzle,
            "solution_grid": shifted_solution,
            "metadata": metadata
        }
        
        processed_puzzles.append(puzzle_entry)
    
    # Save processed puzzles
    with open(args.output, 'w') as f:
        json.dump(processed_puzzles, f, indent=2)
    
    print(f"✅ Saved {len(processed_puzzles)} puzzles with solutions to {args.output}")
    print(f"📊 Solved: {solved_count}, Failed: {failed_count}")
    
    # Save individual metadata files
    metadata_dir = args.output.replace('.json', '_metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    for puzzle_entry in processed_puzzles:
        puzzle_id = puzzle_entry["id"]
        metadata_file = os.path.join(metadata_dir, f"{puzzle_id}_metadata.json")
        
        # Create separate metadata file with puzzle and solution info
        separate_metadata = {
            "puzzle_id": puzzle_id,
            "puzzle_grid": puzzle_entry["puzzle_grid"],
            "solution_grid": puzzle_entry["solution_grid"],
            "metadata": puzzle_entry["metadata"]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(separate_metadata, f, indent=2)
    
    print(f"✅ Saved individual metadata files to {metadata_dir}/")
    
    # Generate images if requested
    if args.regen_images:
        if torchvision is None:
            print("torchvision is not installed. Cannot generate images.")
            return
        
        if args.images_dir is None:
            print("Please specify --images-dir to save generated images.")
            return
        
        print("🖼️ Generating MNIST images...")
        os.makedirs(args.images_dir, exist_ok=True)
        mnist_images = load_mnist_images_1_5(args.mnist_root)
        
        for puzzle_entry in processed_puzzles:
            puzzle_id = puzzle_entry["id"]
            
            # Generate puzzle image
            puzzle_img = make_mnist_grid(puzzle_entry["puzzle_grid"], mnist_images)
            Image.fromarray(puzzle_img).save(os.path.join(args.images_dir, f"{puzzle_id}_puzzle.png"))
            
            # Generate solution image
            solution_img = make_mnist_grid(puzzle_entry["solution_grid"], mnist_images)
            Image.fromarray(solution_img).save(os.path.join(args.images_dir, f"{puzzle_id}_solution.png"))
        
        print(f"✅ Generated MNIST images saved to {args.images_dir}")

if __name__ == '__main__':
    main() 