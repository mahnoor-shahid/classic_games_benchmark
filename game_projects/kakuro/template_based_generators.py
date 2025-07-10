"""
Kakuro Template-Based Generator
Generates Kakuro puzzles using predefined templates and MNIST digit integration
"""

import numpy as np
import random
import time
from typing import Dict, List, Optional, Tuple
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import cv2

class TemplateBasedGenerator:
    def __init__(self):
        # Initialize templates for different difficulties
        self.templates = {
            'easy': self._create_easy_templates(),
            'moderate': self._create_moderate_templates(),
            'hard': self._create_hard_templates()
        }
        
        # Load MNIST images
        self.mnist_images = self._load_mnist_images()
        
        # Initialize statistics
        self.stats = {
            'generated': 0,
            'failed': 0,
            'total_time': 0
        }
        
        print("✅ Template-based generator initialized with templates:")
        for difficulty, templates in self.templates.items():
            print(f"  - {difficulty}: {len(templates)} templates")

    def _load_mnist_images(self) -> Dict[int, List[np.ndarray]]:
        """Load MNIST images for digits 1-9"""
        try:
            print("📥 Loading real MNIST images from dataset...")
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            
            # Normalize and reshape images
            X = X.astype('float32') / 255.0
            X = X.reshape(-1, 28, 28)
            
            # Group images by digit
            digit_images = {}
            for digit in range(1, 10):  # We only need digits 1-9
                digit_images[digit] = X[y == str(digit)]
            
            print("✅ Real MNIST images loaded for digits 1-9")
            return digit_images
            
        except Exception as e:
            print(f"❌ Error loading real MNIST images: {str(e)}")
            print("🔄 Falling back to synthetic patterns...")
            return self._create_synthetic_patterns()

    def _create_synthetic_patterns(self) -> Dict[int, List[np.ndarray]]:
        """Create synthetic patterns for digits when MNIST is not available"""
        patterns = {}
        for digit in range(1, 10):
            patterns[digit] = []
            # Create 10 variations of each digit
            for _ in range(10):
                img = np.zeros((28, 28), dtype=np.float32)
                # Add some random noise and basic shape
                img += np.random.normal(0, 0.1, (28, 28))
                # Add digit-specific features
                if digit == 1:
                    img[:, 13:15] = 0.8
                elif digit == 2:
                    img[5:10, 5:20] = 0.8
                    img[15:20, 5:20] = 0.8
                # Add more digit patterns...
                patterns[digit].append(img)
        return patterns

    def _create_easy_templates(self) -> List[Dict]:
        """Create templates for easy puzzles"""
        templates = []
        
        # Cross pattern
        templates.append({
            'name': 'cross',
            'pattern': np.array([
                [0, 0, 0, 0, 0],
                [0, -1, -1, -1, 0],
                [0, -1, -1, -1, 0],
                [0, -1, -1, -1, 0],
                [0, 0, 0, 0, 0]
            ]),
            'size': 5
        })
        
        # Border pattern
        templates.append({
            'name': 'border',
            'pattern': np.array([
                [0, 0, 0, 0, 0],
                [0, -1, -1, -1, 0],
                [0, -1, 0, -1, 0],
                [0, -1, -1, -1, 0],
                [0, 0, 0, 0, 0]
            ]),
            'size': 5
        })
        
        # Checkerboard pattern
        templates.append({
            'name': 'checkerboard',
            'pattern': np.array([
                [0, 0, 0, 0, 0],
                [0, -1, 0, -1, 0],
                [0, 0, -1, 0, 0],
                [0, -1, 0, -1, 0],
                [0, 0, 0, 0, 0]
            ]),
            'size': 5
        })
        
        return templates

    def _create_moderate_templates(self) -> List[Dict]:
        """Create templates for moderate puzzles"""
        templates = []
        
        # Diamond pattern
        templates.append({
            'name': 'diamond',
            'pattern': np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, -1, -1, -1, 0, 0],
                [0, -1, -1, -1, -1, -1, 0],
                [0, 0, -1, -1, -1, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ]),
            'size': 7
        })
        
        # Spiral pattern
        templates.append({
            'name': 'spiral',
            'pattern': np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, -1, -1, -1, -1, -1, 0],
                [0, -1, 0, 0, 0, -1, 0],
                [0, -1, 0, -1, 0, -1, 0],
                [0, -1, 0, 0, 0, -1, 0],
                [0, -1, -1, -1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ]),
            'size': 7
        })
        
        return templates

    def _create_hard_templates(self) -> List[Dict]:
        """Create templates for hard puzzles"""
        templates = []
        
        # Complex pattern
        templates.append({
            'name': 'complex',
            'pattern': np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, -1, 0, -1, 0, -1, -1, 0],
                [0, -1, 0, -1, 0, -1, 0, -1, 0],
                [0, 0, -1, -1, -1, -1, -1, 0, 0],
                [0, -1, 0, -1, 0, -1, 0, -1, 0],
                [0, -1, -1, 0, -1, 0, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'size': 9
        })
        
        return templates

    def generate_puzzles(self, difficulty: str, count: int) -> List[Dict]:
        """Generate a specified number of puzzles for a given difficulty"""
        puzzles = []
        start_time = time.time()
        
        for i in range(count):
            try:
                puzzle = self._generate_single_puzzle(difficulty)
                if puzzle:
                    puzzles.append(puzzle)
                    print(f"✅ Generated puzzle {i+1}/{count}")
                else:
                    print(f"❌ Failed to generate puzzle {i+1}/{count}")
                    self.stats['failed'] += 1
            except Exception as e:
                print(f"❌ Error generating puzzle {i+1}/{count}: {str(e)}")
                self.stats['failed'] += 1
        
        self.stats['generated'] = len(puzzles)
        self.stats['total_time'] = time.time() - start_time
        return puzzles

    def _generate_single_puzzle(self, difficulty: str) -> Optional[Dict]:
        """Generate a single puzzle for the given difficulty"""
        # Select a random template
        template = random.choice(self.templates[difficulty])
        
        # Create the grid
        grid = template['pattern'].copy()
        
        # Add sums to the grid
        grid = self._add_sums_to_grid(grid, difficulty)
        
        # Select required strategies
        required_strategies = self._select_required_strategies(difficulty)
        
        return {
            'grid': grid.tolist(),
            'template': template['name'],
            'difficulty': difficulty,
            'required_strategies': required_strategies
        }

    def _add_sums_to_grid(self, grid: np.ndarray, difficulty: str) -> np.ndarray:
        """Add sums to the grid based on difficulty"""
        rows, cols = grid.shape
        
        # Define sum ranges based on difficulty
        if difficulty == 'easy':
            min_sum = 3
            max_sum = 9
            min_cells = 2
            max_cells = 4
        elif difficulty == 'moderate':
            min_sum = 4
            max_sum = 12
            min_cells = 3
            max_cells = 5
        else:  # hard
            min_sum = 5
            max_sum = 15
            min_cells = 4
            max_cells = 6
        
        # Add horizontal sums
        for i in range(rows):
            j = 0
            while j < cols:
                if grid[i, j] == -1:
                    # Count consecutive cells
                    count = 0
                    while j + count < cols and grid[i, j + count] == -1:
                        count += 1
                    
                    if count >= min_cells:
                        # Generate a valid sum
                        sum_value = self._generate_valid_sum(count, min_sum, max_sum)
                        grid[i, j-1] = sum_value
                
                j += 1
        
        # Add vertical sums
        for j in range(cols):
            i = 0
            while i < rows:
                if grid[i, j] == -1:
                    # Count consecutive cells
                    count = 0
                    while i + count < rows and grid[i + count, j] == -1:
                        count += 1
                    
                    if count >= min_cells:
                        # Generate a valid sum
                        sum_value = self._generate_valid_sum(count, min_sum, max_sum)
                        grid[i-1, j] = sum_value
                
                i += 1
        
        return grid

    def _generate_valid_sum(self, num_cells: int, min_sum: int, max_sum: int) -> int:
        """Generate a valid sum for a given number of cells"""
        # Calculate minimum possible sum
        min_possible = sum(range(1, num_cells + 1))
        # Calculate maximum possible sum
        max_possible = sum(range(10 - num_cells, 10))
        
        # Adjust min_sum and max_sum based on possible values
        min_sum = max(min_sum, min_possible)
        max_sum = min(max_sum, max_possible)
        
        return random.randint(min_sum, max_sum)

    def _select_required_strategies(self, difficulty: str) -> List[str]:
        """Select required strategies based on difficulty"""
        if difficulty == 'easy':
            basic_strategies = ['single_cell_sum', 'unique_sum_combination', 'cross_reference']
            advanced_strategies = ['sum_partition', 'digit_frequency']
        elif difficulty == 'moderate':
            basic_strategies = ['sum_partition', 'digit_frequency']
            advanced_strategies = ['sum_difference', 'minimum_maximum', 'sum_completion']
        else:  # hard
            basic_strategies = ['sum_completion', 'digit_elimination']
            advanced_strategies = ['sum_difference', 'minimum_maximum', 'cross_reference']
        
        # Select at least one basic strategy
        strategies = [random.choice(basic_strategies)]
        
        # Add some advanced strategies
        num_advanced = random.randint(1, 3)
        strategies.extend(random.sample(advanced_strategies, num_advanced))
        
        return strategies

    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return self.stats 