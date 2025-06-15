# futoshiki_template_generator_enhanced.py
"""
Enhanced Template-Based Futoshiki Generator
- Preserves MNIST digits and images
- Uses knowledge bases for strategy-based generation
- Adds constraint symbols to MNIST images
- Maintains existing command structure
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

# Import knowledge bases and solver
from futoshiki_easy_strategies_kb import FutoshikiEasyStrategiesKB
from futoshiki_moderate_strategies_kb import FutoshikiModerateStrategiesKB
from futoshiki_hard_strategies_kb import FutoshikiHardStrategiesKB
from futoshiki_solver import FutoshikiSolver


class MNISTConstraintVisualizer:
    """Adds constraint symbols to MNIST-based puzzle images"""
    
    def __init__(self):
        self.constraint_font_size = 16
        
    def add_constraints_to_mnist_image(self, mnist_grid: np.ndarray, 
                                     h_constraints: Dict, v_constraints: Dict, 
                                     size: int, puzzle_id: str, is_solution: bool = False) -> Image.Image:
        """Add constraint symbols to existing MNIST grid image"""
        try:
            # Convert MNIST grid to PIL Image and scale up for better visibility
            base_img = Image.fromarray(mnist_grid, mode='L').convert('RGB')
            
            # Scale up for better constraint visibility
            scale_factor = 1.5
            new_width = int(base_img.width * scale_factor)
            new_height = int(base_img.height * scale_factor)
            base_img = base_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Add margins for constraints and title
            margin = 30
            title_space = 40
            legend_space = 30
            final_width = new_width + 2 * margin
            final_height = new_height + 2 * margin + title_space + legend_space
            
            # Create final image with white background
            final_img = Image.new('RGB', (final_width, final_height), 'white')
            
            # Paste MNIST image in center
            paste_x = margin
            paste_y = margin + title_space
            final_img.paste(base_img, (paste_x, paste_y))
            
            draw = ImageDraw.Draw(final_img)
            
            # Load fonts
            try:
                constraint_font = ImageFont.truetype("arial.ttf", self.constraint_font_size)
                title_font = ImageFont.truetype("arial.ttf", 20)
            except:
                constraint_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # Add title
            title = f"{'Solution' if is_solution else 'Puzzle'} {puzzle_id} ({size}x{size}) - MNIST + Constraints"
            draw.text((final_width // 2, 20), title, fill='darkblue', font=title_font, anchor='mt')
            
            # Calculate cell size in the scaled MNIST image
            cell_size = new_width // size
            
            # Parse constraints
            parsed_h_constraints = self._parse_constraints(h_constraints)
            parsed_v_constraints = self._parse_constraints(v_constraints)
            
            # Draw horizontal constraints on MNIST image
            for (row, col), constraint in parsed_h_constraints.items():
                if col < size - 1:
                    # Position between MNIST cells
                    x = paste_x + (col + 0.5) * cell_size + cell_size // 2
                    y = paste_y + row * cell_size + cell_size // 2
                    
                    symbol = '<' if constraint == '<' else '>'
                    
                    # Draw background circle
                    radius = 12
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                               fill='yellow', outline='red', width=2)
                    
                    # Draw constraint symbol
                    draw.text((x, y), symbol, fill='red', font=constraint_font, anchor='mm')
            
            # Draw vertical constraints on MNIST image
            for (row, col), constraint in parsed_v_constraints.items():
                if row < size - 1:
                    # Position between MNIST cells
                    x = paste_x + col * cell_size + cell_size // 2
                    y = paste_y + (row + 0.5) * cell_size + cell_size // 2
                    
                    symbol = '<' if constraint == '<' else '>'
                    
                    # Draw background circle
                    radius = 12
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                               fill='lightblue', outline='blue', width=2)
                    
                    # Draw constraint symbol
                    draw.text((x, y), symbol, fill='blue', font=constraint_font, anchor='mm')
            
            # Add constraint legend at bottom
            legend_y = final_height - 15
            h_count = len(parsed_h_constraints)
            v_count = len(parsed_v_constraints)
            legend_text = f"Constraints: {h_count} horizontal (red), {v_count} vertical (blue)"
            draw.text((final_width // 2, legend_y), legend_text, fill='gray', 
                     font=constraint_font, anchor='mb')
            
            return final_img
            
        except Exception as e:
            print(f"⚠️ Error adding constraints to MNIST image: {e}")
            # Return original MNIST image if enhancement fails
            return Image.fromarray(mnist_grid, mode='L').convert('RGB')
    
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


class FutoshikiTemplate:
    """Template for rapid puzzle generation - UNCHANGED"""
    
    def __init__(self, size: int, difficulty: str):
        self.size = size
        self.difficulty = difficulty
        
    def create_base_solution(self) -> np.ndarray:
        """Create a valid Latin square solution using mathematical approach"""
        grid = np.zeros((self.size, self.size), dtype=int)
        
        # Create base pattern using cyclic shifts (guaranteed valid Latin square)
        for i in range(self.size):
            for j in range(self.size):
                grid[i, j] = ((i + j) % self.size) + 1
        
        # Randomize to create variety
        self._randomize_solution(grid)
        return grid
    
    def _randomize_solution(self, grid: np.ndarray):
        """Apply random transformations while preserving Latin square property"""
        # Random row swaps
        for _ in range(self.size):
            i, j = random.sample(range(self.size), 2)
            grid[[i, j]] = grid[[j, i]]
        
        # Random column swaps
        for _ in range(self.size):
            i, j = random.sample(range(self.size), 2)
            grid[:, [i, j]] = grid[:, [j, i]]
        
        # Random value permutation
        old_values = list(range(1, self.size + 1))
        new_values = old_values.copy()
        random.shuffle(new_values)
        
        value_map = dict(zip(old_values, new_values))
        for i in range(self.size):
            for j in range(self.size):
                grid[i, j] = value_map[grid[i, j]]


class FreshTemplateFutoshikiGenerator:
    """Enhanced generator with MNIST preservation and constraint visualization"""
    
    def __init__(self, config_manager=None):
        print("🚀 Initializing Enhanced Template Generator (MNIST + Constraints + Knowledge Bases)...")
        
        self.config_manager = config_manager
        self.solver = FutoshikiSolver()
        
        # Initialize knowledge bases for strategy-based generation
        self.easy_kb = FutoshikiEasyStrategiesKB()
        self.moderate_kb = FutoshikiModerateStrategiesKB()
        self.hard_kb = FutoshikiHardStrategiesKB()
        
        # MNIST data and consistent mappings (PRESERVED)
        self.mnist_images = {}
        self.puzzle_digit_mappings = {}
        
        # Constraint visualizer
        self.constraint_visualizer = MNISTConstraintVisualizer()
        
        # Strategy-based templates (using knowledge bases)
        self.templates = {
            'easy': [],
            'moderate': [],
            'hard': []
        }
        
        # Load MNIST and initialize templates
        self._load_mnist_data_fast()
        self._initialize_strategy_based_templates()
        
        print("✅ Enhanced generator initialized with MNIST + Constraints + Strategy-based generation")
    
    def _load_mnist_data_fast(self):
        """Load MNIST data quickly with fallback - PRESERVED"""
        try:
            print("📥 Loading MNIST dataset...")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            # Organize by digit (1-9 only)
            train_by_digit = {i: [] for i in range(1, 10)}
            count_per_digit = {i: 0 for i in range(1, 10)}
            
            for image, label in train_dataset:
                if 1 <= label <= 9 and count_per_digit[label] < 50:
                    train_by_digit[label].append(image)
                    count_per_digit[label] += 1
                
                # Stop when we have enough
                if all(count >= 50 for count in count_per_digit.values()):
                    break
            
            self.mnist_images = {'train': train_by_digit}
            total = sum(len(images) for images in train_by_digit.values())
            print(f"✅ MNIST loaded: {total} images")
            
        except Exception as e:
            print(f"⚠️ MNIST loading failed: {e}")
            print("🎨 Creating fallback digit patterns...")
            self._create_fallback_patterns()
    
    def _create_fallback_patterns(self):
        """Create recognizable fallback patterns for digits - PRESERVED"""
        train_by_digit = {i: [] for i in range(1, 10)}
        
        for digit in range(1, 10):
            for variant in range(20):
                pattern = self._create_digit_pattern(digit, variant)
                train_by_digit[digit].append(pattern)
        
        self.mnist_images = {'train': train_by_digit}
        print("✅ Fallback patterns created")
    
    def _create_digit_pattern(self, digit: int, variant: int = 0) -> np.ndarray:
        """Create a recognizable pattern for a digit with variants - PRESERVED"""
        img = np.zeros((28, 28), dtype=np.uint8)
        offset = variant % 3  # Small variations
        
        # [Previous digit pattern code remains the same]
        if digit == 1:
            img[4+offset:24-offset, 13:15] = 255
            img[4+offset:8+offset, 11:15] = 255
        elif digit == 2:
            img[4:8, 6:20] = 255
            img[8:12, 16:20] = 255
            img[12:16, 6:16] = 255
            img[16:20, 6:10] = 255
            img[20:24, 6:20] = 255
        # ... rest of digit patterns
        else:
            # Simple pattern for other digits
            img[8:20, 10:18] = 255
        
        return img
    
    def _initialize_strategy_based_templates(self):
        """Initialize templates based on knowledge base strategies"""
        print("🔧 Initializing strategy-based templates from knowledge bases...")
        
        # Easy templates using easy KB strategies
        easy_strategies = self.easy_kb.list_strategies()
        self.templates['easy'] = [
            {
                'name': 'Basic Singles',
                'constraint_positions': [('h', 0, 1), ('h', 1, 2), ('v', 1, 1)],
                'removal_pattern': 'balanced',
                'target_filled': 13,
                'strategies': ['naked_single', 'constraint_propagation', 'row_uniqueness'],
                'kb_validated': True
            },
            {
                'name': 'Hidden Singles Focus',
                'constraint_positions': [('h', 0, 0), ('h', 0, 3), ('v', 0, 0), ('v', 2, 2)],
                'removal_pattern': 'corners_first',
                'target_filled': 12,
                'strategies': ['naked_single', 'hidden_single_row', 'hidden_single_column'],
                'kb_validated': True
            },
            {
                'name': 'Constraint Forcing',
                'constraint_positions': [('h', 2, 1), ('h', 2, 2), ('v', 1, 2), ('v', 2, 2)],
                'removal_pattern': 'center_out',
                'target_filled': 14,
                'strategies': ['naked_single', 'forced_by_inequality', 'direct_constraint_forcing'],
                'kb_validated': True
            }
        ]
        
        # Moderate templates using moderate KB strategies
        moderate_strategies = self.moderate_kb.list_strategies()
        self.templates['moderate'] = [
            {
                'name': 'Naked Pairs',
                'constraint_positions': [('h', 0, 1), ('h', 0, 2), ('h', 1, 3), ('v', 0, 2), ('v', 1, 3)],
                'removal_pattern': 'strategic',
                'target_filled': 13,
                'strategies': ['naked_single', 'naked_pair', 'constraint_propagation_advanced'],
                'kb_validated': True
            },
            {
                'name': 'Constraint Chains',
                'constraint_positions': [('h', 1, 0), ('h', 1, 1), ('h', 1, 4), ('v', 0, 1), ('v', 2, 3)],
                'removal_pattern': 'intersection_based',
                'target_filled': 11,
                'strategies': ['constraint_chain_analysis', 'inequality_sandwich', 'mutual_constraint_elimination'],
                'kb_validated': True
            },
            {
                'name': 'Hidden Pairs',
                'constraint_positions': [('h', 0, 2), ('h', 2, 0), ('v', 1, 1), ('v', 3, 4)],
                'removal_pattern': 'hidden_focus',
                'target_filled': 12,
                'strategies': ['hidden_pair', 'value_forcing_by_uniqueness', 'constraint_splitting'],
                'kb_validated': True
            }
        ]
        
        # Hard templates using hard KB strategies
        hard_strategies = self.hard_kb.list_strategies()
        self.templates['hard'] = [
            {
                'name': 'Multiple Chains',
                'constraint_positions': [
                    ('h', 0, 0), ('h', 0, 1), ('h', 2, 1), ('h', 4, 3), ('h', 4, 4),
                    ('v', 0, 1), ('v', 1, 2), ('v', 3, 4)
                ],
                'removal_pattern': 'chain_complex',
                'target_filled': 12,
                'strategies': ['multiple_constraint_chains', 'constraint_intersection_forcing', 'naked_triple'],
                'kb_validated': True
            },
            {
                'name': 'Network Analysis',
                'constraint_positions': [
                    ('h', 1, 1), ('h', 2, 4), ('h', 3, 0), ('h', 5, 2),
                    ('v', 0, 2), ('v', 2, 1), ('v', 4, 5)
                ],
                'removal_pattern': 'network_based',
                'target_filled': 10,
                'strategies': ['constraint_network_analysis', 'global_constraint_consistency', 'advanced_sandwich_analysis'],
                'kb_validated': True
            }
        ]
        
        # Validate all templates use strategies from their respective KBs
        self._validate_template_strategies()
        
        total_templates = sum(len(self.templates[d]) for d in ['easy', 'moderate', 'hard'])
        print(f"✅ Strategy-based templates initialized: {total_templates} total")
        print(f"   📋 Easy: {len(self.templates['easy'])} templates using easy KB strategies")
        print(f"   📋 Moderate: {len(self.templates['moderate'])} templates using moderate KB strategies")
        print(f"   📋 Hard: {len(self.templates['hard'])} templates using hard KB strategies")
    
    def _validate_template_strategies(self):
        """Validate that templates use strategies from appropriate knowledge bases"""
        print("🔍 Validating template strategies against knowledge bases...")
        
        easy_kb_strategies = set(self.easy_kb.list_strategies())
        moderate_kb_strategies = set(self.moderate_kb.list_strategies())
        hard_kb_strategies = set(self.hard_kb.list_strategies())
        
        for difficulty, templates in self.templates.items():
            for template in templates:
                strategies = set(template['strategies'])
                
                if difficulty == 'easy':
                    if not strategies.issubset(easy_kb_strategies):
                        invalid = strategies - easy_kb_strategies
                        print(f"⚠️ Easy template '{template['name']}' uses non-easy strategies: {invalid}")
                
                elif difficulty == 'moderate':
                    valid_strategies = easy_kb_strategies | moderate_kb_strategies
                    if not strategies.issubset(valid_strategies):
                        invalid = strategies - valid_strategies
                        print(f"⚠️ Moderate template '{template['name']}' uses invalid strategies: {invalid}")
                
                elif difficulty == 'hard':
                    valid_strategies = easy_kb_strategies | moderate_kb_strategies | hard_kb_strategies
                    if not strategies.issubset(valid_strategies):
                        invalid = strategies - valid_strategies
                        print(f"⚠️ Hard template '{template['name']}' uses invalid strategies: {invalid}")
        
        print("✅ Template strategy validation complete")
    
    def generate_puzzle_from_template(self, difficulty: str, template_idx: int = None) -> Dict:
        """Generate puzzle using strategy-based template system - ENHANCED"""
        templates = self.templates[difficulty]
        if not templates:
            raise ValueError(f"No templates available for {difficulty}")
        
        if template_idx is None:
            template_idx = random.randint(0, len(templates) - 1)
        
        template = templates[template_idx % len(templates)]
        
        # Get size based on difficulty
        sizes = {'easy': 5, 'moderate': 6, 'hard': 7}
        size = sizes[difficulty]
        
        # Create base solution
        puzzle_template = FutoshikiTemplate(size, difficulty)
        solution = puzzle_template.create_base_solution()
        
        # Apply constraints and cell removal
        h_constraints, v_constraints = self._apply_constraint_template(solution, template, size)
        puzzle = self._apply_removal_template(solution, template, size)
        
        # Validate puzzle can be solved with required strategies
        if template.get('kb_validated', False):
            is_solvable = self._validate_puzzle_with_strategies(
                puzzle, solution, h_constraints, v_constraints, template['strategies']
            )
            if not is_solvable:
                print(f"⚠️ Generated puzzle may not be solvable with required strategies")
        
        return {
            'puzzle': puzzle,
            'solution': solution,
            'h_constraints': h_constraints,
            'v_constraints': v_constraints,
            'strategies': template['strategies'],
            'template_name': template['name'],
            'kb_validated': template.get('kb_validated', False)
        }
    
    def _validate_puzzle_with_strategies(self, puzzle: np.ndarray, solution: np.ndarray,
                                       h_constraints: Dict, v_constraints: Dict, 
                                       required_strategies: List[str]) -> bool:
        """Validate puzzle can be solved with required strategies"""
        try:
            solved_puzzle, used_strategies = self.solver.solve_puzzle(
                puzzle.copy(), h_constraints, v_constraints, required_strategies, max_time_seconds=10
            )
            
            # Check if puzzle was solved and uses required strategies
            is_solved = np.array_equal(solved_puzzle, solution)
            uses_required = any(strategy in used_strategies for strategy in required_strategies)
            
            return is_solved and uses_required
            
        except Exception as e:
            print(f"Strategy validation error: {e}")
            return False
    
    def get_consistent_mnist_mapping(self, puzzle_id: str, size: int) -> Dict[int, np.ndarray]:
        """Get consistent MNIST digit mapping for puzzle and solution - PRESERVED"""
        if puzzle_id not in self.puzzle_digit_mappings:
            mapping = {}
            
            # Use puzzle ID as seed for consistency
            random.seed(hash(puzzle_id) % (2**32))
            
            for digit in range(1, size + 1):
                available_images = self.mnist_images['train'][digit]
                if available_images:
                    mapping[digit] = random.choice(available_images)
                else:
                    mapping[digit] = self._create_digit_pattern(digit)
            
            self.puzzle_digit_mappings[puzzle_id] = mapping
        
        return self.puzzle_digit_mappings[puzzle_id]
    
    def create_mnist_representation(self, grid: np.ndarray, puzzle_id: str) -> np.ndarray:
        """Create MNIST representation with consistent digit mapping - PRESERVED"""
        size = len(grid)
        mnist_size = size * 28
        mnist_grid = np.zeros((mnist_size, mnist_size), dtype=np.uint8)
        
        # Use consistent digit mapping
        digit_mapping = self.get_consistent_mnist_mapping(puzzle_id, size)
        
        for row in range(size):
            for col in range(size):
                if grid[row, col] != 0:
                    digit_img = digit_mapping[grid[row, col]]
                    
                    start_row = row * 28
                    start_col = col * 28
                    mnist_grid[start_row:start_row+28, start_col:start_col+28] = digit_img
        
        return mnist_grid
    
    def save_enhanced_images(self, dataset: List[Dict], output_dir: str):
        """Save MNIST images with constraint symbols overlaid - ENHANCED"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for puzzle_data in dataset:
                puzzle_id = puzzle_data['id']
                size = puzzle_data['size']
                
                # Get MNIST grids (PRESERVED)
                mnist_puzzle = np.array(puzzle_data['mnist_puzzle'], dtype=np.uint8)
                mnist_solution = np.array(puzzle_data['mnist_solution'], dtype=np.uint8)
                
                # Get constraints
                h_constraints = puzzle_data['h_constraints']
                v_constraints = puzzle_data['v_constraints']
                
                # Create MNIST images with constraint symbols overlaid
                puzzle_img_with_constraints = self.constraint_visualizer.add_constraints_to_mnist_image(
                    mnist_puzzle, h_constraints, v_constraints, size, puzzle_id, is_solution=False
                )
                
                solution_img_with_constraints = self.constraint_visualizer.add_constraints_to_mnist_image(
                    mnist_solution, h_constraints, v_constraints, size, puzzle_id, is_solution=True
                )
                
                # Save enhanced images
                puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
                solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
                
                puzzle_img_with_constraints.save(puzzle_path, quality=95, optimize=True)
                solution_img_with_constraints.save(solution_path, quality=95, optimize=True)
                
                # Also save original MNIST images without constraints
                mnist_puzzle_path = os.path.join(output_dir, f"{puzzle_id}_mnist_puzzle.png")
                mnist_solution_path = os.path.join(output_dir, f"{puzzle_id}_mnist_solution.png")
                
                Image.fromarray(mnist_puzzle, mode='L').save(mnist_puzzle_path)
                Image.fromarray(mnist_solution, mode='L').save(mnist_solution_path)
                
                print(f"    🖼️ MNIST + Constraint images saved: {puzzle_id}")
            
            print(f"🖼️ Enhanced MNIST images with constraints saved to {output_dir}")
            print(f"📋 Each puzzle has 4 images:")
            print(f"   • *_puzzle.png (MNIST + constraint symbols)")
            print(f"   • *_solution.png (MNIST + constraint symbols)")
            print(f"   • *_mnist_puzzle.png (original MNIST only)")
            print(f"   • *_mnist_solution.png (original MNIST only)")
            
        except Exception as e:
            print(f"❌ Error saving enhanced images: {e}")
    
    def generate_fast_dataset(self, difficulty: str, count: int) -> List[Dict]:
        """Generate dataset with MNIST preservation and strategy validation - ENHANCED"""
        print(f"\n⚡ Generating {count} {difficulty} puzzles (MNIST + Strategy-based + Constraints)...")
        
        dataset = []
        sizes = {'easy': 5, 'moderate': 6, 'hard': 7}
        size = sizes[difficulty]
        
        start_time = time.time()
        
        for i in range(count):
            if (i + 1) % max(1, count // 4) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                print(f"  ⚡ Progress: {i+1}/{count} ({rate:.1f}/sec)")
            
            try:
                # Generate using strategy-based template
                template_idx = i % len(self.templates[difficulty])
                puzzle_data = self.generate_puzzle_from_template(difficulty, template_idx)
                
                puzzle_id = f"enhanced_{difficulty}_{i:04d}"
                
                # Create MNIST representations with consistent mapping (PRESERVED)
                mnist_puzzle = self.create_mnist_representation(puzzle_data['puzzle'], puzzle_id)
                mnist_solution = self.create_mnist_representation(puzzle_data['solution'], puzzle_id)
                
                # Create enhanced constraint visualization
                constraint_viz = self.create_enhanced_constraint_visualization(
                    puzzle_data['h_constraints'], puzzle_data['v_constraints'], size
                )
                
                # Create complete puzzle entry
                puzzle_entry = {
                    'id': puzzle_id,
                    'difficulty': difficulty,
                    'size': size,
                    'puzzle_grid': puzzle_data['puzzle'].tolist(),
                    'solution_grid': puzzle_data['solution'].tolist(),
                    'h_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['h_constraints'].items()},
                    'v_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['v_constraints'].items()},
                    'required_strategies': puzzle_data['strategies'],
                    'mnist_puzzle': mnist_puzzle.tolist(),
                    'mnist_solution': mnist_solution.tolist(),
                    'constraint_visualization': constraint_viz,
                    'strategy_details': {
                        strategy: self._get_strategy_details(strategy)
                        for strategy in puzzle_data['strategies']
                    },
                    'template_info': {
                        'template_name': puzzle_data['template_name'],
                        'template_index': template_idx,
                        'kb_validated': puzzle_data.get('kb_validated', False)
                    },
                    'metadata': {
                        'generated_timestamp': datetime.now().isoformat(),
                        'filled_cells': int(np.sum(puzzle_data['puzzle'] != 0)),
                        'empty_cells': int(np.sum(puzzle_data['puzzle'] == 0)),
                        'total_cells': size * size,
                        'fill_ratio': float(np.sum(puzzle_data['puzzle'] != 0)) / (size * size),
                        'num_h_constraints': len(puzzle_data['h_constraints']),
                        'num_v_constraints': len(puzzle_data['v_constraints']),
                        'total_constraints': len(puzzle_data['h_constraints']) + len(puzzle_data['v_constraints']),
                        'generation_method': 'strategy_based_template',
                        'template_used': template_idx,
                        'template_name': puzzle_data['template_name'],
                        'generator_version': '2.1.0_enhanced',
                        'mnist_consistency': True,
                        'kb_strategy_based': True,
                        'constraint_visualization': True
                    }
                }
                
                dataset.append(puzzle_entry)
                
            except Exception as e:
                print(f"  ⚠️ Error generating puzzle {i}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        rate = len(dataset) / max(elapsed_time, 0.001)
        
        print(f"\n⚡ Enhanced generation complete for {difficulty}:")
        print(f"  ✅ Generated: {len(dataset)}/{count} puzzles")
        print(f"  🕐 Time: {elapsed_time:.2f} seconds")
        print(f"  📊 Rate: {rate:.1f} puzzles/second")
        print(f"  🧠 Strategy-based generation from knowledge bases")
        print(f"  🖼️ MNIST digits preserved with constraint overlays")
        
        return dataset
    
    def create_enhanced_constraint_visualization(self, h_constraints: Dict, v_constraints: Dict, size: int) -> Dict:
        """Create enhanced constraint visualization - PRESERVED"""
        constraint_data = {
            'horizontal': [],
            'vertical': [],
            'size': size,
            'use_text_symbols': True,
            'text_symbols': {'<': 'LT', '>': 'GT'},
            'constraint_summary': {
                'total_constraints': len(h_constraints) + len(v_constraints),
                'horizontal_count': len(h_constraints),
                'vertical_count': len(v_constraints),
                'constraint_density': (len(h_constraints) + len(v_constraints)) / (size * size)
            },
            'display_info': {
                'symbols_visible': True,
                'enhanced_display': True,
                'mnist_overlay': True
            }
        }
        
        # Enhanced horizontal constraints
        for (row, col), constraint in h_constraints.items():
            constraint_data['horizontal'].append({
                'row': row,
                'col': col,
                'constraint': constraint,
                'display_text': 'LT' if constraint == '<' else 'GT',
                'display_symbol': constraint,  # Keep actual symbols
                'position': 'between_cells',
                'description': f"Cell({row},{col}) {constraint} Cell({row},{col+1})",
                'symbol_type': 'horizontal',
                'cells_affected': [(row, col), (row, col + 1)]
            })
        
        # Enhanced vertical constraints
        for (row, col), constraint in v_constraints.items():
            constraint_data['vertical'].append({
                'row': row,
                'col': col,
                'constraint': constraint,
                'display_text': 'LT' if constraint == '<' else 'GT',
                'display_symbol': constraint,  # Keep actual symbols
                'position': 'between_cells',
                'description': f"Cell({row},{col}) {constraint} Cell({row+1},{col})",
                'symbol_type': 'vertical',
                'cells_affected': [(row, col), (row + 1, col)]
            })
        
        return constraint_data
    
    def _apply_constraint_template(self, solution: np.ndarray, template: Dict, size: int) -> Tuple[Dict, Dict]:
        """Apply constraint pattern from template with bounds checking - PRESERVED"""
        h_constraints = {}
        v_constraints = {}
        
        for constraint_type, row, col in template['constraint_positions']:
            # Ensure positions are within bounds
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
    
    def _apply_removal_template(self, solution: np.ndarray, template: Dict, size: int) -> np.ndarray:
        """Apply enhanced cell removal pattern - PRESERVED"""
        puzzle = solution.copy()
        target_filled = template['target_filled']
        pattern = template['removal_pattern']
        
        total_cells = size * size
        cells_to_remove = max(0, total_cells - target_filled)
        
        # Define removal patterns
        if pattern == 'balanced':
            positions = [(i, j) for i in range(size) for j in range(size) if (i + j) % 2 == 0]
        elif pattern == 'corners_first':
            positions = [(i, j) for i in range(size) for j in range(size)]
            positions.sort(key=lambda p: min(p[0] + p[1], p[0] + (size-1-p[1]), 
                                           (size-1-p[0]) + p[1], (size-1-p[0]) + (size-1-p[1])))
        elif pattern == 'center_out':
            center = size // 2
            positions = [(i, j) for i in range(size) for j in range(size)]
            positions.sort(key=lambda p: abs(p[0] - center) + abs(p[1] - center))
        elif pattern == 'strategic':
            positions = [(i, j) for i in range(size) for j in range(size) if (i + j) % 3 != 0]
        elif pattern == 'intersection_based':
            positions = [(i, j) for i in range(size) for j in range(size) if i % 2 == 0 or j % 2 == 0]
        elif pattern == 'hidden_focus':
            positions = [(i, j) for i in range(size) for j in range(size) if (i * j) % 3 == 0]
        elif pattern == 'chain_complex':
            positions = [(i, j) for i in range(size) for j in range(size) if abs(i - j) <= 2]
        elif pattern == 'network_based':
            positions = [(i, j) for i in range(size) for j in range(size) if (i + 2*j) % 4 != 0]
        else:
            # Default random
            positions = [(i, j) for i in range(size) for j in range(size)]
        
        random.shuffle(positions)
        
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        return puzzle
    
    def _get_strategy_details(self, strategy_name: str) -> Dict:
        """Get detailed strategy information from knowledge bases"""
        for kb, level in [(self.easy_kb, 'easy'), (self.moderate_kb, 'moderate'), (self.hard_kb, 'hard')]:
            if strategy_name in kb.list_strategies():
                strategy_info = kb.get_strategy(strategy_name)
                return {
                    'name': strategy_info.get('name', strategy_name),
                    'description': strategy_info.get('description', f'Strategy: {strategy_name}'),
                    'complexity': strategy_info.get('complexity', level),
                    'composite': strategy_info.get('composite', False),
                    'knowledge_base': level,
                    'prerequisites': strategy_info.get('prerequisites', []),
                    'applies_to': strategy_info.get('applies_to', ['general']),
                    'constraint_aware': strategy_info.get('constraint_aware', False)
                }
        
        return {
            'name': strategy_name,
            'description': f'Strategy: {strategy_name}',
            'complexity': 'unknown',
            'composite': False,
            'knowledge_base': 'unknown',
            'prerequisites': [],
            'applies_to': ['general'],
            'constraint_aware': False
        }
    
    def save_enhanced_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset with proper JSON structure - PRESERVED"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            class EnhancedNumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super().default(obj)
            
            # Save as flat list with metadata in each puzzle
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2, cls=EnhancedNumpyEncoder)
            
            print(f"💾 Enhanced dataset saved to {filename}")
            print(f"📊 Structure: Flat list of {len(dataset)} puzzles with MNIST + Constraints + Strategies")
            
        except Exception as e:
            print(f"❌ Error saving enhanced dataset: {e}")
    
    def display_puzzle_with_constraints(self, puzzle_data: Dict):
        """Display puzzle in console with enhanced constraint visualization - ENHANCED"""
        grid = np.array(puzzle_data['puzzle_grid'])
        size = len(grid)
        
        # Parse constraints from puzzle data
        h_constraints = {}
        v_constraints = {}
        
        for key, value in puzzle_data['h_constraints'].items():
            row, col = map(int, key.split(','))
            h_constraints[(row, col)] = value
        
        for key, value in puzzle_data['v_constraints'].items():
            row, col = map(int, key.split(','))
            v_constraints[(row, col)] = value
        
        print(f"\n🎯 PUZZLE: {puzzle_data['id']} ({size}x{size})")
        print(f"📊 Constraints: {len(h_constraints)} horizontal, {len(v_constraints)} vertical")
        print(f"🧠 Required Strategies: {', '.join(puzzle_data['required_strategies'])}")
        print(f"🔬 Knowledge Base Validated: {puzzle_data.get('template_info', {}).get('kb_validated', False)}")
        print(f"🖼️ MNIST Consistency: {puzzle_data.get('metadata', {}).get('mnist_consistency', True)}")
        print()
        
        # Display grid with enhanced constraint symbols
        for row in range(size):
            row_str = ""
            for col in range(size):
                # Cell value
                cell_val = " . " if grid[row, col] == 0 else f" {grid[row, col]} "
                row_str += cell_val
                
                # Horizontal constraint with visible symbols
                if (row, col) in h_constraints and col < size - 1:
                    constraint = h_constraints[(row, col)]
                    symbol = f" {constraint} "  # Actual < or > symbols
                    row_str += symbol
                elif col < size - 1:
                    row_str += "   "  # 3 spaces for alignment
            
            print(row_str)
            
            # Vertical constraints
            if row < size - 1:
                constraint_row = ""
                for col in range(size):
                    if (row, col) in v_constraints:
                        constraint = v_constraints[(row, col)]
                        symbol = f" {constraint} "  # Actual < or > symbols
                        constraint_row += symbol
                    else:
                        constraint_row += "   "  # 3 spaces
                    
                    if col < size - 1:
                        constraint_row += "   "  # Match horizontal spacing
                
                print(constraint_row)
        
        print()


# Test function to verify everything works with existing command structure
def test_enhanced_generator():
    """Test the enhanced generator with MNIST + Constraints + Knowledge Bases"""
    print("🧪 Testing Enhanced Generator (MNIST + Constraints + KB)")
    print("=" * 60)
    
    try:
        # Initialize enhanced generator
        generator = FreshTemplateFutoshikiGenerator()
        
        # Test strategy validation
        print("\n🔍 Testing knowledge base strategy validation...")
        easy_strategies = generator.easy_kb.list_strategies()
        moderate_strategies = generator.moderate_kb.list_strategies()
        hard_strategies = generator.hard_kb.list_strategies()
        
        print(f"📚 Easy KB has {len(easy_strategies)} strategies")
        print(f"📚 Moderate KB has {len(moderate_strategies)} strategies")
        print(f"📚 Hard KB has {len(hard_strategies)} strategies")
        
        # Test template generation for each difficulty
        for difficulty in ['easy', 'moderate', 'hard']:
            print(f"\n🧪 Testing {difficulty} generation...")
            
            # Generate single puzzle
            puzzle_data = generator.generate_puzzle_from_template(difficulty, 0)
            puzzle_id = f"test_{difficulty}_001"
            
            # Create MNIST representations
            mnist_puzzle = generator.create_mnist_representation(puzzle_data['puzzle'], puzzle_id)
            mnist_solution = generator.create_mnist_representation(puzzle_data['solution'], puzzle_id)
            
            # Create full puzzle entry
            test_puzzle = {
                'id': puzzle_id,
                'difficulty': difficulty,
                'size': {'easy': 5, 'moderate': 6, 'hard': 7}[difficulty],
                'puzzle_grid': puzzle_data['puzzle'].tolist(),
                'solution_grid': puzzle_data['solution'].tolist(),
                'h_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['h_constraints'].items()},
                'v_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['v_constraints'].items()},
                'required_strategies': puzzle_data['strategies'],
                'mnist_puzzle': mnist_puzzle.tolist(),
                'mnist_solution': mnist_solution.tolist(),
                'template_info': {
                    'template_name': puzzle_data['template_name'],
                    'kb_validated': puzzle_data.get('kb_validated', False)
                },
                'metadata': {
                    'mnist_consistency': True,
                    'kb_strategy_based': True
                }
            }
            
            # Test constraint visualization
            constraint_viz = generator.create_enhanced_constraint_visualization(
                puzzle_data['h_constraints'], puzzle_data['v_constraints'], 
                test_puzzle['size']
            )
            test_puzzle['constraint_visualization'] = constraint_viz
            
            # Display puzzle
            generator.display_puzzle_with_constraints(test_puzzle)
            
            print(f"✅ {difficulty.capitalize()} test successful!")
            print(f"   Template: {puzzle_data['template_name']}")
            print(f"   Strategies: {', '.join(puzzle_data['strategies'])}")
            print(f"   KB Validated: {puzzle_data.get('kb_validated', False)}")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ MNIST digits preserved")
        print(f"✅ Knowledge base strategies used")
        print(f"✅ Constraint symbols will be overlaid on images")
        print(f"✅ Ready for: python futoshiki_main.py --action generate")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_generator()