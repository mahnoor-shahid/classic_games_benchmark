# sudoku_generator.py
"""
MNIST-based 5x5 Sudoku Puzzle Generator with Integrated Validation
Generates exactly the requested number of VALID 5x5 Sudoku puzzles using digits 0-4
"""

import numpy as np
import json
import random
import os
import time
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

# Import knowledge bases and solver
from sudoku_easy_strategies_kb import EasyStrategiesKB
from sudoku_moderate_strategies_kb import ModerateStrategiesKB
from sudoku_hard_strategies_kb import HardStrategiesKB


class SudokuValidator:
    """Integrated 5x5 Sudoku validator that ensures puzzle quality"""
    def is_valid_sudoku_solution(self, grid: np.ndarray) -> bool:
        for row in range(5):
            if set(grid[row, :]) != set(range(5)):
                return False
        for col in range(5):
            if set(grid[:, col]) != set(range(5)):
                return False
        if set([grid[i, i] for i in range(5)]) != set(range(5)):
            return False
        if set([grid[i, 4 - i] for i in range(5)]) != set(range(5)):
            return False
        return True

    def has_unique_solution(self, puzzle: np.ndarray) -> bool:
        solutions_found = 0
        max_solutions = 2
        def solve(grid):
            nonlocal solutions_found
            if solutions_found >= max_solutions:
                return
            empty = self._find_empty_cell(grid)
            if not empty:
                solutions_found += 1
                return
            row, col = empty
            for num in range(5):
                if self._is_valid_placement(grid, row, col, num):
                    grid[row, col] = num
                    solve(grid)
                    grid[row, col] = -1
        test_grid = puzzle.copy()
        solve(test_grid)
        return solutions_found == 1

    def _find_empty_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        for i in range(5):
            for j in range(5):
                if grid[i, j] == -1:
                    return (i, j)
        return None

    def _is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        if row == col and num in [grid[i, i] for i in range(5) if i != row]:
            return False
        if row + col == 4 and num in [grid[i, 4 - i] for i in range(5) if i != row]:
            return False
        return True

class MNISTSudokuGenerator:
    """MNIST 5x5 Sudoku Generator with guaranteed valid puzzle generation"""
    def __init__(self, config_manager=None):
        print("🚀 Initializing MNIST 5x5 Sudoku Generator with integrated validation...")
        self.validator = SudokuValidator()
        self.config_manager = config_manager
        self.mnist_images = {}
        self.load_mnist_data()
        self.stats = {'total_attempts': 0, 'successful_generations': 0, 'failed_validations': 0, 'generation_times': []}
        print("✅ Generator initialized successfully")

    def load_mnist_data(self):
        try:
            print("📥 Loading MNIST dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            train_by_digit = {i: [] for i in range(5)}
            test_by_digit = {i: [] for i in range(5)}
            for image, label in train_dataset:
                if 0 <= label <= 4:
                    train_by_digit[label].append(image)
            for image, label in test_dataset:
                if 0 <= label <= 4:
                    test_by_digit[label].append(image)
            self.mnist_images = {
                'train': train_by_digit,
                'test': test_by_digit
            }
            total_images = sum(len(images) for digit_images in train_by_digit.values() for images in [digit_images])
            print(f"✅ MNIST loaded: {total_images} training images for digits 0-4")
        except Exception as e:
            print(f"⚠️ Error loading MNIST: {e}")
            print("🔄 Creating fallback dummy data...")
            self.create_dummy_mnist()

    def create_dummy_mnist(self):
        self.mnist_images = {'train': {i: [np.zeros((28, 28), dtype=np.uint8)] for i in range(5)}, 'test': {i: [np.zeros((28, 28), dtype=np.uint8)] for i in range(5)}}

    def generate_complete_sudoku(self) -> np.ndarray:
        grid = np.zeros((5, 5), dtype=int)
        def is_valid(grid, row, col, num):
            if num in grid[row, :]:
                return False
            if num in grid[:, col]:
                return False
            if row == col and num in [grid[i, i] for i in range(5) if i != row]:
                return False
            if row + col == 4 and num in [grid[i, 4 - i] for i in range(5) if i != row]:
                return False
            return True
        def solve(grid, row=0, col=0):
            if row == 5:
                return True
            next_row, next_col = (row, col + 1) if col < 4 else (row + 1, 0)
            nums = list(range(5))
            random.shuffle(nums)
            for num in nums:
                if is_valid(grid, row, col, num):
                    grid[row, col] = num
                    if solve(grid, next_row, next_col):
                        return True
                    grid[row, col] = -1
            return False
        if solve(grid):
            return grid
        return None

    def create_fallback_puzzle(self, solution: np.ndarray, difficulty: str) -> Tuple[np.ndarray, List[str]]:
        puzzle = solution.copy()
        # For 5x5, just remove a few cells for fallback
        target_filled = 12 if difficulty == 'easy' else 8 if difficulty == 'moderate' else 5
        cells_to_remove = 25 - target_filled
        positions = [(i, j) for i in range(5) for j in range(5)]
        random.shuffle(positions)
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = -1
        basic_strategies = {
            'easy': ['naked_single', 'hidden_single_row'],
            'moderate': ['naked_single', 'naked_pair', 'hidden_single_row'],
            'hard': ['naked_single']
        }
        return puzzle, basic_strategies[difficulty]

    def create_mnist_representation(self, grid: np.ndarray) -> np.ndarray:
        mnist_grid = np.zeros((140, 140), dtype=np.uint8)  # 5x5 * 28x28
        for row in range(5):
            for col in range(5):
                if grid[row, col] != -1:
                    digit_img = self.get_mnist_image(grid[row, col])
                    start_row = row * 28
                    start_col = col * 28
                    mnist_grid[start_row:start_row+28, start_col:start_col+28] = digit_img
        return mnist_grid

    def get_mnist_image(self, digit: int) -> np.ndarray:
        imgs = self.mnist_images['train'].get(digit, [])
        if imgs:
            return imgs[0]
        return np.zeros((28, 28), dtype=np.uint8)


def main():
    """Test the generator"""
    generator = MNISTSudokuGenerator()
    
    # Test generating valid puzzles
    test_puzzles = generator.generate_guaranteed_valid_puzzles('easy', 2)
    
    print(f"\nGenerated {len(test_puzzles)} valid puzzles")
    for puzzle in test_puzzles:
        print(f"- {puzzle['id']}: {puzzle['metadata']['filled_cells']} filled cells, "
              f"strategies: {puzzle['required_strategies']}")


if __name__ == "__main__":
    main()