# complete_futoshiki_with_constraints.py
"""
Complete Futoshiki Generator with Visible Constraint Symbols
This script generates puzzles with constraint symbols (< and >) visible directly on the images
"""

import numpy as np
import json
import random
import os
import time
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
import torchvision.transforms as transforms

# Import existing modules
from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from futoshiki_hard_strategies_kb import FutoshikiHardStrategiesKB
from futoshiki_solver import FutoshikiSolver


class ConstraintVisualizer:
    """Handles adding constraint symbols directly to puzzle images"""
    
    def __init__(self, cell_size: int = 80):
        self.cell_size = cell_size
        self.constraint_font_size = 24
        self.digit_font_size = 40
        self.margin = 50
        
    def create_puzzle_image_with_constraints(self, puzzle_grid: List[List[int]], 
                                           solution_grid: List[List[int]], 
                                           h_constraints: Dict, v_constraints: Dict, 
                                           size: int, puzzle_id: str, is_solution: bool = False) -> Image.Image:
        """Create a beautiful puzzle image with visible constraint symbols"""
        
        # Calculate image dimensions
        grid_size = size * self.cell_size
        img_width = grid_size + 2 * self.margin
        img_height = grid_size + 2 * self.margin + 80  # Extra space for title and legend
        
        # Create white background
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            digit_font = ImageFont.truetype("arial.ttf", self.digit_font_size)
            constraint_font = ImageFont.truetype("arial.ttf", self.constraint_font_size)
            title_font = ImageFont.truetype("arial.ttf", 28)
        except:
            digit_font = ImageFont.load_default()
            constraint_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        title = f"{'Solution' if is_solution else 'Puzzle'} {puzzle_id} ({size}x{size})"
        draw.text((img_width // 2, 20), title, fill='darkblue', font=title_font, anchor='mt')
        
        # Calculate grid offset
        grid_offset_x = self.margin
        grid_offset_y = self.margin + 50
        
        # Draw grid background
        for row in range(size):
            for col in range(size):
                x = grid_offset_x + col * self.cell_size
                y = grid_offset_y + row * self.cell_size
                
                # Alternate cell colors for better visibility
                cell_color = 'white' if (row + col) % 2 == 0 else '#f8f8f8'
                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size], 
                             fill=cell_color, outline='lightgray')
        
        # Draw grid lines
        for i in range(size + 1):
            # Vertical lines
            x = grid_offset_x + i * self.cell_size
            draw.line([(x, grid_offset_y), (x, grid_offset_y + grid_size)], 
                     fill='black', width=3)
            
            # Horizontal lines
            y = grid_offset_y + i * self.cell_size
            draw.line([(grid_offset_x, y), (grid_offset_x + grid_size, y)], 
                     fill='black', width=3)
        
        # Draw numbers
        grid_to_use = solution_grid if is_solution else puzzle_grid
        for row in range(size):
            for col in range(size):
                if grid_to_use[row][col] != 0:
                    x = grid_offset_x + col * self.cell_size + self.cell_size // 2
                    y = grid_offset_y + row * self.cell_size + self.cell_size // 2
                    
                    number = str(grid_to_use[row][col])
                    color = 'darkgreen' if is_solution else 'darkblue'
                    
                    # Add shadow effect
                    draw.text((x + 2, y + 2), number, fill='lightgray', font=digit_font, anchor='mm')
                    draw.text((x, y), number, fill=color, font=digit_font, anchor='mm')
        
        # Parse constraints
        parsed_h_constraints = self._parse_constraints(h_constraints)
        parsed_v_constraints = self._parse_constraints(v_constraints)
        
        # Draw horizontal constraints
        for (row, col), constraint in parsed_h_constraints.items():
            if col < size - 1:
                x = grid_offset_x + col * self.cell_size + self.cell_size
                y = grid_offset_y + row * self.cell_size + self.cell_size // 2
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw constraint background
                radius = 18
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                           fill='yellow', outline='red', width=3)
                
                # Draw constraint symbol
                draw.text((x, y), symbol, fill='red', font=constraint_font, anchor='mm')
        
        # Draw vertical constraints
        for (row, col), constraint in parsed_v_constraints.items():
            if row < size - 1:
                x = grid_offset_x + col * self.cell_size + self.cell_size // 2
                y = grid_offset_y + row * self.cell_size + self.cell_size
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw constraint background
                radius = 18
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                           fill='lightblue', outline='blue', width=3)
                
                # Draw constraint symbol
                draw.text((x, y), symbol, fill='blue', font=constraint_font, anchor='mm')
        
        # Add constraint legend
        legend_y = img_height - 30
        h_count = len(parsed_h_constraints)
        v_count = len(parsed_v_constraints)
        legend_text = f"Constraints: {h_count} horizontal (red), {v_count} vertical (blue)"
        draw.text((img_width // 2, legend_y), legend_text, fill='gray', 
                 font=constraint_font, anchor='mb')
        
        return img
    
    def _parse_constraints(self, constraints: Dict) -> Dict:
        """Parse constraint dictionary handling both string and tuple keys"""
        parsed = {}
        for key, value in constraints.items():
            if isinstance(key, str):
                row, col = map(int, key.split(','))
                parsed[(row, col)] = value
            else:
                parsed[key] = value
        return parsed


class EnhancedFutoshikiGenerator:
    """Enhanced Futoshiki generator with beautiful constraint visualization"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Futoshiki Generator with Constraint Visualization...")
        
        # Initialize components
        self.solver = FutoshikiSolver()
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        
        # Initialize constraint visualizer
        self.visualizer = ConstraintVisualizer()
        
        # MNIST data
        self.mnist_images = {}
        self.puzzle_digit_mappings = {}
        
        # Load MNIST and templates
        self._load_mnist_data()
        self._initialize_templates()
        
        print("✅ Enhanced generator initialized with constraint visualization!")
    
    def _load_mnist_data(self):
        """Load MNIST data or create fallback patterns"""
        try:
            print("📥 Loading MNIST dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            train_by_digit = {i: [] for i in range(1, 10)}
            count_per_digit = {i: 0 for i in range(1, 10)}
            
            for image, label in train_dataset:
                if 1 <= label <= 9 and count_per_digit[label] < 30:
                    train_by_digit[label].append(image)
                    count_per_digit[label] += 1
                
                if all(count >= 30 for count in count_per_digit.values()):
                    break
            
            self.mnist_images = {'train': train_by_digit}
            print(f"✅ MNIST loaded successfully")
            
        except Exception as e:
            print(f"⚠️ MNIST loading failed: {e}")
            print("🎨 Creating fallback patterns...")
            self._create_fallback_patterns()
    
    def _create_fallback_patterns(self):
        """Create fallback digit patterns"""
        train_by_digit = {i: [] for i in range(1, 10)}
        
        for digit in range(1, 10):
            for _ in range(10):
                pattern = self._create_digit_pattern(digit)
                train_by_digit[digit].append(pattern)
        
        self.mnist_images = {'train': train_by_digit}
        print("✅ Fallback patterns created")
    
    def _create_digit_pattern(self, digit: int) -> np.ndarray:
        """Create simple digit pattern"""
        img = np.zeros((28, 28), dtype=np.uint8)
        # Simple patterns for each digit
        if digit == 1:
            img[6:22, 13:15] = 255
        elif digit == 2:
            img[6:10, 8:20] = 255
            img[10:14, 16:20] = 255
            img[14:18, 8:16] = 255
            img[18:22, 8:20] = 255
        # Add more patterns as needed...
        else:
            # Simple rectangle for other digits
            img[8:20, 10:18] = 255
        
        return img
    
    def _initialize_templates(self):
        """Initialize puzzle templates"""
        self.templates = {
            'easy': [
                {
                    'name': 'Simple Constraints',
                    'constraint_positions': [('h', 0, 1), ('h', 1, 2), ('v', 1, 1)],
                    'target_filled': 13,
                    'strategies': ['naked_single', 'constraint_propagation']
                },
                {
                    'name': 'Corner Focus',
                    'constraint_positions': [('h', 0, 0), ('v', 0, 0), ('h', 3, 3)],
                    'target_filled': 12,
                    'strategies': ['naked_single', 'row_uniqueness']
                }
            ],
            'moderate': [
                {
                    'name': 'Chain Pattern',
                    'constraint_positions': [('h', 0, 1), ('h', 0, 2), ('v', 1, 2), ('v', 2, 1)],
                    'target_filled': 11,
                    'strategies': ['naked_pair', 'constraint_chain_analysis']
                }
            ],
            'hard': [
                {
                    'name': 'Complex Network',
                    'constraint_positions': [('h', 0, 0), ('h', 1, 1), ('v', 0, 1), ('v', 1, 2), ('h', 2, 3)],
                    'target_filled': 10,
                    'strategies': ['multiple_constraint_chains', 'naked_triple']
                }
            ]
        }
    
    def generate_puzzle_with_constraints(self, difficulty: str = 'easy') -> Dict:
        """Generate a single puzzle with enhanced constraint visualization"""
        sizes = {'easy': 5, 'moderate': 6, 'hard': 7}
        size = sizes[difficulty]
        
        # Create base solution (Latin square)
        solution = self._create_latin_square(size)
        
        # Get template
        templates = self.templates[difficulty]
        template = random.choice(templates)
        
        # Apply constraints
        h_constraints, v_constraints = self._apply_constraints_from_template(solution, template, size)
        
        # Create puzzle by removing cells
        puzzle = self._create_puzzle_from_solution(solution, template, size)
        
        # Generate unique puzzle ID
        puzzle_id = f"enhanced_{difficulty}_{int(time.time() % 10000):04d}"
        
        return {
            'id': puzzle_id,
            'difficulty': difficulty,
            'size': size,
            'puzzle_grid': puzzle.tolist(),
            'solution_grid': solution.tolist(),
            'h_constraints': {f"{k[0]},{k[1]}": v for k, v in h_constraints.items()},
            'v_constraints': {f"{k[0]},{k[1]}": v for k, v in v_constraints.items()},
            'strategies': template['strategies'],
            'template_name': template['name']
        }
    
    def _create_latin_square(self, size: int) -> np.ndarray:
        """Create a valid Latin square"""
        grid = np.zeros((size, size), dtype=int)
        
        # Simple Latin square generation
        for i in range(size):
            for j in range(size):
                grid[i, j] = ((i + j) % size) + 1
        
        # Randomize
        for _ in range(size * 2):
            i, j = random.sample(range(size), 2)
            grid[[i, j]] = grid[[j, i]]  # Swap rows
        
        for _ in range(size * 2):
            i, j = random.sample(range(size), 2)
            grid[:, [i, j]] = grid[:, [j, i]]  # Swap columns
        
        return grid
    
    def _apply_constraints_from_template(self, solution: np.ndarray, template: Dict, size: int) -> Tuple[Dict, Dict]:
        """Apply constraints based on template"""
        h_constraints = {}
        v_constraints = {}
        
        for constraint_type, row, col in template['constraint_positions']:
            if constraint_type == 'h' and 0 <= row < size and 0 <= col < size - 1:
                left_val = solution[row, col]
                right_val = solution[row, col + 1]
                if left_val != right_val:
                    constraint = '<' if left_val < right_val else '>'
                    h_constraints[(row, col)] = constraint
            
            elif constraint_type == 'v' and 0 <= row < size - 1 and 0 <= col < size:
                top_val = solution[row, col]
                bottom_val = solution[row + 1, col]
                if top_val != bottom_val:
                    constraint = '<' if top_val < bottom_val else '>'
                    v_constraints[(row, col)] = constraint
        
        return h_constraints, v_constraints
    
    def _create_puzzle_from_solution(self, solution: np.ndarray, template: Dict, size: int) -> np.ndarray:
        """Create puzzle by removing cells strategically"""
        puzzle = solution.copy()
        total_cells = size * size
        target_filled = template['target_filled']
        cells_to_remove = total_cells - target_filled
        
        # Get all positions and shuffle
        positions = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(positions)
        
        # Remove cells
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        return puzzle
    
    def save_puzzle_images_with_constraints(self, puzzle_data: Dict, output_dir: str):
        """Save beautiful puzzle images with visible constraint symbols"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            puzzle_id = puzzle_data['id']
            puzzle_grid = puzzle_data['puzzle_grid']
            solution_grid = puzzle_data['solution_grid']
            h_constraints = puzzle_data['h_constraints']
            v_constraints = puzzle_data['v_constraints']
            size = puzzle_data['size']
            
            # Create puzzle image with constraints
            puzzle_img = self.visualizer.create_puzzle_image_with_constraints(
                puzzle_grid, solution_grid, h_constraints, v_constraints, 
                size, puzzle_id, is_solution=False
            )
            
            # Create solution image with constraints
            solution_img = self.visualizer.create_puzzle_image_with_constraints(
                puzzle_grid, solution_grid, h_constraints, v_constraints, 
                size, puzzle_id, is_solution=True
            )
            
            # Save images
            puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
            solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
            
            puzzle_img.save(puzzle_path, quality=95, optimize=True)
            solution_img.save(solution_path, quality=95, optimize=True)
            
            print(f"🖼️ Images with visible constraints saved: {puzzle_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving images: {e}")
            return False
    
    def generate_dataset_with_constraint_images(self, difficulty: str = 'easy', count: int = 5):
        """Generate a complete dataset with constraint visualization"""
        print(f"\n🎯 Generating {count} {difficulty} puzzles with constraint visualization...")
        
        dataset = []
        output_dir = f"./enhanced_futoshiki_{difficulty}"
        
        start_time = time.time()
        
        for i in range(count):
            try:
                # Generate puzzle
                puzzle_data = self.generate_puzzle_with_constraints(difficulty)
                
                # Save images with constraints
                success = self.save_puzzle_images_with_constraints(puzzle_data, output_dir)
                
                if success:
                    dataset.append(puzzle_data)
                    print(f"  ✅ Generated puzzle {i+1}/{count}: {puzzle_data['id']}")
                else:
                    print(f"  ⚠️ Failed to save images for puzzle {i+1}")
                
            except Exception as e:
                print(f"  ❌ Error generating puzzle {i+1}: {e}")
        
        # Save dataset JSON
        if dataset:
            dataset_path = os.path.join(output_dir, f"enhanced_dataset_{difficulty}.json")
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            elapsed = time.time() - start_time
            print(f"\n🏆 Generated {len(dataset)} {difficulty} puzzles in {elapsed:.1f}s")
            print(f"📁 Images saved to: {output_dir}")
            print(f"📄 Dataset saved to: {dataset_path}")
            print(f"🖼️ Each puzzle has constraint symbols (< and >) visible on the images!")
        
        return dataset


def main():
    """Main function to generate puzzles with visible constraints"""
    print("=" * 70)
    print("🎯 ENHANCED FUTOSHIKI GENERATOR WITH VISIBLE CONSTRAINTS")
    print("=" * 70)
    print("Features:")
    print("  🖼️ Constraint symbols (< and >) visible directly on puzzle images")
    print("  🎨 Beautiful grid layout with alternating cell colors")
    print("  📐 Multiple puzzle sizes: Easy 5x5, Moderate 6x6, Hard 7x7")
    print("  🎯 Template-based fast generation")
    print("=" * 70)
    
    try:
        # Initialize generator
        generator = EnhancedFutoshikiGenerator()
        
        # Generate samples for each difficulty
        for difficulty in ['easy', 'moderate', 'hard']:
            print(f"\n🎮 Generating {difficulty} puzzles...")
            dataset = generator.generate_dataset_with_constraint_images(difficulty, 3)
            
            if dataset:
                print(f"✅ Successfully generated {len(dataset)} {difficulty} puzzles")
                print(f"🔍 Check the './enhanced_futoshiki_{difficulty}' folder for images")
                
                # Display first puzzle info
                first_puzzle = dataset[0]
                print(f"\n📋 Sample {difficulty} puzzle: {first_puzzle['id']}")
                print(f"   Size: {first_puzzle['size']}x{first_puzzle['size']}")
                print(f"   Constraints: {len(first_puzzle['h_constraints'])} horizontal, {len(first_puzzle['v_constraints'])} vertical")
                print(f"   Strategies: {', '.join(first_puzzle['strategies'])}")
            else:
                print(f"❌ Failed to generate {difficulty} puzzles")
        
        print(f"\n🎉 GENERATION COMPLETE!")
        print(f"🖼️ All puzzle images now have visible constraint symbols!")
        print(f"📁 Check the generated folders for your enhanced puzzle images")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()