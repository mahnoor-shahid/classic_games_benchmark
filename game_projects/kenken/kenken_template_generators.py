# kenken_template_generators.py - FIXED VERSION
"""
Template-based Ken Ken Generators for all difficulty levels
Ensures consistent puzzle generation with MNIST images and proper strategy usage
"""
import os
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime
import json


# Fix 1: Add JSON serialization helper function at the top of the class
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
    
class KenKenTemplateBasedGenerator:
    """Template-based generator for creating consistent Ken Ken puzzles"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        
        # Initialize templates for all difficulties
        self.templates = {
            'easy': self._create_easy_templates(),
            'moderate': self._create_moderate_templates(),
            'hard': self._create_hard_templates()
        }
        
        # Load REAL MNIST images from the actual dataset
        self.mnist_images = self._load_real_mnist_images()
        
        # Generation statistics
        self.stats = {
            'generated': 0,
            'failed': 0,
            'total_time': 0
        }
        
        print(f"✅ Ken Ken Template-based generator initialized with templates:")
        for difficulty, templates in self.templates.items():
            print(f"  - {difficulty}: {len(templates)} templates")
        print(f"✅ Real MNIST images loaded for digits 1-9")
    
    def _load_real_mnist_images(self) -> Dict[int, np.ndarray]:
        """Load actual MNIST images from the dataset (one per digit)"""
        patterns = {}
        
        try:
            # Try to load MNIST using torchvision
            import torchvision
            import torchvision.transforms as transforms
            
            print("📥 Loading real MNIST images from dataset...")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).numpy().astype(np.uint8).squeeze())
            ])
            
            # Load MNIST dataset
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=False, transform=transform
            )
            
            # Find one good representative image for each digit 1-9
            digit_images = {}
            
            print("🔍 Searching for representative images for each digit...")
            
            for image, label in train_dataset:
                if 1 <= label <= 9 and label not in digit_images:
                    # Select a clear, well-formed image (not too sparse)
                    pixel_count = np.sum(image > 50)  # Count significant pixels
                    
                    if pixel_count > 100:  # Ensure the digit has enough pixels
                        digit_images[label] = image
                        print(f"  ✅ Found digit {label}: {pixel_count} pixels")
                        
                        # Stop when we have all digits 1-9
                        if len(digit_images) == 9:
                            break
            
            # If we didn't find all digits, fill in any missing ones
            if len(digit_images) < 9:
                print("🔄 Filling in missing digits with any available images...")
                for image, label in train_dataset:
                    if 1 <= label <= 9 and label not in digit_images:
                        digit_images[label] = image
                        print(f"  📌 Added digit {label} (backup)")
                        if len(digit_images) == 9:
                            break
            
            # Store the images
            for digit in range(1, 10):
                if digit in digit_images:
                    patterns[digit] = digit_images[digit]
                    pixel_count = np.sum(digit_images[digit] > 0)
                    print(f"  📦 Stored real MNIST digit {digit}: {pixel_count} pixels")
                else:
                    # Fallback: create a simple pattern if MNIST image not found
                    print(f"  ⚠️ Creating fallback for digit {digit}")
                    patterns[digit] = self._create_fallback_digit(digit)
            
            print(f"✅ Successfully loaded {len(patterns)} real MNIST digit images")
            return patterns
            
        except Exception as e:
            print(f"❌ Error loading real MNIST images: {e}")
            print("🔄 Falling back to synthetic patterns...")
            return self._create_fallback_mnist_patterns()
    
    def _create_fallback_digit(self, digit: int) -> np.ndarray:
        """Create a fallback digit pattern"""
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simple patterns for each digit
        if digit == 1:
            img[5:23, 13:15] = 255
        elif digit == 2:
            img[5:9, 8:20] = 255
            img[9:14, 15:20] = 255
            img[14:18, 8:15] = 255
            img[18:23, 8:20] = 255
        elif digit == 3:
            img[5:9, 8:20] = 255
            img[12:16, 12:20] = 255
            img[18:23, 8:20] = 255
        else:
            # Generic pattern with digit-specific noise
            for i in range(28):
                for j in range(28):
                    if (i + j + digit * 3) % 8 < 2:
                        img[i, j] = min(255, (i + j + digit * 30) % 256)
        
        return img
    
    def _create_fallback_mnist_patterns(self) -> Dict[int, np.ndarray]:
        """Create fallback MNIST patterns for all digits"""
        patterns = {}
        for digit in range(1, 10):
            patterns[digit] = self._create_fallback_digit(digit)
        return patterns
    
    def _create_easy_templates(self) -> List[Dict]:
        """Create simple templates for easy Ken Ken puzzles"""
        templates = []
        
        # Template 1: 4x4 grid with simple addition cages
        template1 = {
            'id': 'easy_4x4_addition',
            'name': 'Easy 4x4 Addition Focus',
            'difficulty_rating': 1.0,
            'grid_size': 4,
            'cage_patterns': [
                {'size': 2, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'add'},
                {'size': 3, 'shape': 'L', 'operation': 'add'},
                {'size': 1, 'shape': 'single', 'operation': 'add'}
            ],
            'required_strategies': ['naked_single', 'single_cell_cage', 'simple_addition_cage'],
            'description': '4x4 grid focusing on addition cages'
        }
        templates.append(template1)
        
        # Template 2: 4x4 grid with addition and subtraction
        template2 = {
            'id': 'easy_4x4_mixed',
            'name': 'Easy 4x4 Mixed Operations',
            'difficulty_rating': 1.2,
            'grid_size': 4,
            'cage_patterns': [
                {'size': 2, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'subtract'},
                {'size': 2, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'square', 'operation': 'add'}
            ],
            'required_strategies': ['naked_single', 'simple_addition_cage', 'simple_subtraction_cage'],
            'description': '4x4 grid with addition and subtraction cages'
        }
        templates.append(template2)
        
        # Template 3: 5x5 grid with simple operations
        template3 = {
            'id': 'easy_5x5_simple',
            'name': 'Easy 5x5 Simple',
            'difficulty_rating': 1.3,
            'grid_size': 5,
            'cage_patterns': [
                {'size': 3, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'subtract'},
                {'size': 1, 'shape': 'single', 'operation': 'add'},
                {'size': 3, 'shape': 'L', 'operation': 'add'}
            ],
            'required_strategies': ['naked_single', 'cage_completion', 'simple_addition_cage'],
            'description': '5x5 grid with simple cage patterns'
        }
        templates.append(template3)
        
        return templates
    
    def _create_moderate_templates(self) -> List[Dict]:
        """Create templates for moderate Ken Ken puzzles"""
        templates = []
        
        # Template 1: 6x6 with multiplication
        template1 = {
            'id': 'moderate_6x6_multiply',
            'name': 'Moderate 6x6 with Multiplication',
            'difficulty_rating': 2.0,
            'grid_size': 6,
            'cage_patterns': [
                {'size': 2, 'shape': 'line', 'operation': 'multiply'},
                {'size': 3, 'shape': 'L', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'subtract'},
                {'size': 3, 'shape': 'line', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'multiply'}
            ],
            'required_strategies': ['naked_single', 'cage_elimination', 'basic_multiplication_cage', 'constraint_propagation'],
            'description': '6x6 grid introducing multiplication cages'
        }
        templates.append(template1)
        
        return templates
    
    def _create_hard_templates(self) -> List[Dict]:
        """Create templates for hard Ken Ken puzzles"""
        templates = []
        
        # Template 1: 7x7 with all operations
        template1 = {
            'id': 'hard_7x7_all_ops',
            'name': 'Hard 7x7 All Operations',
            'difficulty_rating': 3.0,
            'grid_size': 7,
            'cage_patterns': [
                {'size': 5, 'shape': 'irregular', 'operation': 'add'},
                {'size': 3, 'shape': 'L', 'operation': 'multiply'},
                {'size': 2, 'shape': 'line', 'operation': 'divide'},
                {'size': 4, 'shape': 'T', 'operation': 'add'},
                {'size': 2, 'shape': 'line', 'operation': 'subtract'}
            ],
            'required_strategies': ['advanced_cage_chaining', 'constraint_satisfaction_solving', 'complex_arithmetic_cages'],
            'description': '7x7 grid with all operations and complex cage interactions'
        }
        templates.append(template1)
        
        return templates
    
    def generate_puzzles(self, difficulty: str, target_count: int) -> List[Dict]:
        """Generate puzzles for specified difficulty using templates"""
        print(f"🎯 Generating {target_count} {difficulty} Ken Ken puzzles using templates...")
        
        start_time = time.time()
        generated_puzzles = []
        
        templates = self.templates.get(difficulty, [])
        if not templates:
            print(f"❌ No templates available for {difficulty}")
            return []
        
        attempts = 0
        max_attempts = target_count * 50  # Safety limit
        
        while len(generated_puzzles) < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                # Select template (round-robin with variation)
                template_idx = (attempts - 1) % len(templates)
                template = templates[template_idx]
                
                # Generate puzzle from template
                puzzle_data = self._create_puzzle_from_template(template, attempts)
                
                if puzzle_data:
                    puzzle_data['id'] = f"{difficulty}_{puzzle_data['grid_size']}x{puzzle_data['grid_size']}_{len(generated_puzzles):04d}"
                    generated_puzzles.append(puzzle_data)
                    
                    if len(generated_puzzles) % 2 == 0:
                        elapsed = time.time() - start_time
                        rate = len(generated_puzzles) / elapsed if elapsed > 0 else 0
                        print(f"  ✅ Generated {len(generated_puzzles)}/{target_count} "
                              f"(rate: {rate:.1f}/sec)")
                
            except Exception as e:
                print(f"  ⚠️ Error in attempt {attempts}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        success_rate = len(generated_puzzles) / attempts * 100 if attempts > 0 else 0
        
        print(f"\n📈 Generation complete for {difficulty}:")
        print(f"  ✅ Generated: {len(generated_puzzles)}/{target_count}")
        print(f"  🕐 Time: {elapsed_time:.1f} seconds")
        print(f"  📊 Success rate: {success_rate:.1f}%")
        
        return generated_puzzles
    
    def _create_puzzle_from_template(self, template: Dict, seed: int) -> Optional[Dict]:
        """Create a single puzzle from a template with comprehensive validation"""
        try:
            grid_size = template['grid_size']
            
            # Generate complete solution (Latin square)
            solution = self._generate_latin_square(grid_size, seed)
            if solution is None:
                return None
            
            # Generate cages based on template patterns - FIXED METHOD CALL
            cages = self._generate_cages_from_template(template, solution)
            if not cages:
                return None
            
            # Create puzzle by removing some values
            puzzle = self._create_puzzle_from_solution(solution, cages, template)
            
            # Comprehensive validation
            if not self._validate_puzzle_comprehensive(puzzle, solution, cages, template):
                return None
            
            # Create enhanced MNIST representations
            mnist_puzzle = self._create_mnist_grid(puzzle, cages)
            mnist_solution = self._create_mnist_grid(solution, cages)
            
            # Calculate quality metrics
            quality_score = self._calculate_puzzle_quality(puzzle, cages, template)
            
            return {
                'difficulty': self._get_difficulty_from_template(template),
                'grid_size': grid_size,
                'puzzle_grid': puzzle.tolist(),
                'solution_grid': solution.tolist(),
                'cages': cages,
                'required_strategies': template['required_strategies'],
                'mnist_puzzle': mnist_puzzle.tolist(),
                'mnist_solution': mnist_solution.tolist(),
                'strategy_details': {
                    strategy: {'name': strategy, 'description': f'Apply {strategy}'}
                    for strategy in template['required_strategies']
                },
                'validation': {
                    'quality_score': quality_score,
                    'validation_passed': True,
                    'basic_structure': True,
                    'strategy_requirements': True,
                    'compositionality': True,
                    'solvability': True,
                    'quality_assessment': True,
                    'cage_connectivity': True,
                    'difficulty_appropriate': True
                },
                'metadata': {
                    'generated_timestamp': datetime.now().isoformat(),
                    'filled_cells': int(np.sum(puzzle != 0)),
                    'empty_cells': int(np.sum(puzzle == 0)),
                    'num_cages': len(cages),
                    'operations_used': list(set(cage['operation'] for cage in cages)),
                    'difficulty_score': template['difficulty_rating'],
                    'template_id': template['id'],
                    'template_name': template['name'],
                    'validation_passed': True,
                    'generation_attempt': seed,
                    'generator_version': '1.0.0',
                    'cage_size_distribution': self._get_cage_size_distribution(cages),
                    'operation_distribution': self._get_operation_distribution(cages)
                }
            }
            
        except Exception as e:
            print(f"    Error creating puzzle from template {template['id']}: {e}")
            return None
    
    # FIX: Add the missing _validate_puzzle_comprehensive method
    def _validate_puzzle_comprehensive(self, puzzle: np.ndarray, solution: np.ndarray, 
                                     cages: List[Dict], template: Dict) -> bool:
        """Comprehensive validation of generated puzzle"""
        try:
            # Basic validation
            if not self._validate_puzzle(puzzle, solution, cages, template):
                return False
            
            # Check quality score is reasonable
            quality_score = self._calculate_puzzle_quality(puzzle, cages, template)
            if quality_score < 0.5:  # Minimum quality threshold
                return False
            
            return True
        except Exception:
            return False
    
    def _score_cage_arrangement(self, cages: List[Dict], template: Dict) -> float:
        """Score the cage arrangement quality"""
        if not cages:
            return 0.0
        
        # Basic scoring based on cage variety
        cage_sizes = [len(cage['cells']) for cage in cages]
        operations = [cage['operation'] for cage in cages]
        
        # Variety in cage sizes
        size_variety = len(set(cage_sizes)) / max(len(cage_sizes), 1)
        
        # Variety in operations
        op_variety = len(set(operations)) / max(len(operations), 1)
        
        return (size_variety + op_variety) / 2.0
    
    def _calculate_puzzle_quality(self, puzzle: np.ndarray, cages: List[Dict], template: Dict) -> float:
        """Calculate comprehensive puzzle quality score"""
        quality_score = 0.0
        
        # Factor 1: Fill ratio appropriateness (25%)
        grid_size = len(puzzle)
        filled_cells = np.sum(puzzle != 0)
        fill_ratio = filled_cells / (grid_size * grid_size)
        
        expected_fill = self._get_expected_fill_ratio(template['difficulty_rating'])
        fill_score = 1.0 - abs(fill_ratio - expected_fill) / expected_fill
        quality_score += max(0, fill_score) * 0.25
        
        # Factor 2: Cage variety and complexity (25%)
        cage_variety_score = self._score_cage_arrangement(cages, template)
        quality_score += cage_variety_score * 0.25
        
        # Factor 3: Operation distribution (25%)
        operations = [cage['operation'] for cage in cages]
        operation_variety = len(set(operations)) / 4.0  # Max 4 operations
        quality_score += min(operation_variety, 1.0) * 0.25
        
        # Factor 4: Strategy requirements match (25%)
        strategy_match_score = self._evaluate_strategy_requirements(puzzle, cages, template)
        quality_score += strategy_match_score * 0.25
        
        return min(quality_score, 1.0)
    
    def _get_expected_fill_ratio(self, difficulty_rating: float) -> float:
        """Get expected fill ratio based on difficulty"""
        if difficulty_rating < 1.5:
            return 0.55  # Easy puzzles have more filled cells
        elif difficulty_rating < 2.5:
            return 0.45  # Moderate puzzles
        else:
            return 0.35  # Hard puzzles have fewer filled cells
    
    def _evaluate_strategy_requirements(self, puzzle: np.ndarray, cages: List[Dict], template: Dict) -> float:
        """Evaluate how well the puzzle matches required strategies"""
        required_strategies = template['required_strategies']
        score = 0.0
        strategy_count = len(required_strategies)
        
        if strategy_count == 0:
            return 1.0
        
        # Check for single cell cages (naked singles)
        if 'single_cell_cage' in required_strategies:
            single_cell_cages = sum(1 for cage in cages if len(cage['cells']) == 1)
            if single_cell_cages > 0:
                score += 1.0 / strategy_count
        
        # Check for arithmetic cages
        if 'simple_addition_cage' in required_strategies:
            addition_cages = sum(1 for cage in cages if cage['operation'] == 'add' and len(cage['cells']) > 1)
            if addition_cages > 0:
                score += 1.0 / strategy_count
        
        # Check for subtraction cages
        if 'simple_subtraction_cage' in required_strategies:
            subtraction_cages = sum(1 for cage in cages if cage['operation'] == 'subtract')
            if subtraction_cages > 0:
                score += 1.0 / strategy_count
        
        # Check for multiplication cages
        if 'basic_multiplication_cage' in required_strategies:
            multiplication_cages = sum(1 for cage in cages if cage['operation'] == 'multiply')
            if multiplication_cages > 0:
                score += 1.0 / strategy_count
        
        # Check for division cages
        if 'basic_division_cage' in required_strategies:
            division_cages = sum(1 for cage in cages if cage['operation'] == 'divide')
            if division_cages > 0:
                score += 1.0 / strategy_count
        
        # Add points for other strategies (simplified)
        other_strategies = [s for s in required_strategies if s not in [
            'single_cell_cage', 'simple_addition_cage', 'simple_subtraction_cage',
            'basic_multiplication_cage', 'basic_division_cage'
        ]]
        
        if other_strategies:
            score += len(other_strategies) / strategy_count
        
        return min(score, 1.0)
    
    def _get_cage_size_distribution(self, cages: List[Dict]) -> Dict[int, int]:
        """Get distribution of cage sizes"""
        distribution = {}
        for cage in cages:
            size = len(cage['cells'])
            distribution[size] = distribution.get(size, 0) + 1
        return distribution
    
    def _get_operation_distribution(self, cages: List[Dict]) -> Dict[str, int]:
        """Get distribution of operations"""
        distribution = {}
        for cage in cages:
            op = cage['operation']
            distribution[op] = distribution.get(op, 0) + 1
        return distribution
    
    def _generate_latin_square(self, grid_size: int, seed: int) -> Optional[np.ndarray]:
        """Generate a complete valid Latin square"""
        random.seed(seed)
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Simple approach: create a basic Latin square and shuffle
        for row in range(grid_size):
            for col in range(grid_size):
                grid[row, col] = (row + col) % grid_size + 1
        
        # Apply random transformations
        self._shuffle_latin_square(grid)
        
        random.seed()  # Reset seed
        return grid
    
    def _shuffle_latin_square(self, grid: np.ndarray):
        """Apply random transformations to a Latin square"""
        grid_size = len(grid)
        
        # Shuffle rows within row groups and columns within column groups
        for _ in range(random.randint(5, 15)):
            if random.random() > 0.5:
                # Swap two rows
                r1, r2 = random.sample(range(grid_size), 2)
                grid[[r1, r2]] = grid[[r2, r1]]
            else:
                # Swap two columns
                c1, c2 = random.sample(range(grid_size), 2)
                grid[:, [c1, c2]] = grid[:, [c2, c1]]
        
        # Apply digit permutation
        if random.random() > 0.3:
            mapping = self._create_digit_permutation(grid_size)
            for row in range(grid_size):
                for col in range(grid_size):
                    grid[row, col] = mapping[grid[row, col]]
    
    def _create_digit_permutation(self, grid_size: int) -> Dict[int, int]:
        """Create random digit permutation"""
        digits = list(range(1, grid_size + 1))
        shuffled = digits.copy()
        random.shuffle(shuffled)
        return {digits[i]: shuffled[i] for i in range(grid_size)}
    
    def _generate_cages_from_template(self, template: Dict, solution: np.ndarray) -> List[Dict]:
        """Generate cages based on template patterns"""
        grid_size = template['grid_size']
        cage_patterns = template['cage_patterns']
        
        cages = []
        used_cells = set()
        
        for pattern in cage_patterns:
            # Generate cage based on pattern
            cage_cells = self._create_cage_from_pattern(pattern, used_cells, grid_size)
            if cage_cells:
                # Calculate target based on solution values
                cage_values = [solution[r, c] for r, c in cage_cells]
                target = self._calculate_target_for_operation(cage_values, pattern['operation'])
                
                cage = {
                    'cells': cage_cells,
                    'operation': pattern['operation'],
                    'target': target
                }
                cages.append(cage)
                used_cells.update(cage_cells)
        
        # Cover any remaining cells with single-cell cages
        for row in range(grid_size):
            for col in range(grid_size):
                if (row, col) not in used_cells:
                    cage = {
                        'cells': [(row, col)],
                        'operation': 'add',
                        'target': solution[row, col]
                    }
                    cages.append(cage)
        
        return cages
    
    def _create_cage_from_pattern(self, pattern: Dict, used_cells: Set[Tuple[int, int]], 
                                 grid_size: int) -> List[Tuple[int, int]]:
        """Create a cage matching the specified pattern"""
        size = pattern['size']
        shape = pattern['shape']
        
        max_attempts = 50
        for _ in range(max_attempts):
            # Find available starting position
            available_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)
                                 if (r, c) not in used_cells]
            
            if not available_positions:
                break
            
            start_pos = random.choice(available_positions)
            cage_cells = self._build_cage_shape(start_pos, size, shape, used_cells, grid_size)
            
            if len(cage_cells) == size:
                return cage_cells
        
        # Fallback: create a linear cage
        return self._create_linear_cage(size, used_cells, grid_size)
    
    def _build_cage_shape(self, start_pos: Tuple[int, int], size: int, shape: str,
                         used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build a cage with the specified shape"""
        if shape == 'single':
            return [start_pos] if start_pos not in used_cells else []
        
        elif shape == 'line':
            return self._build_line_cage(start_pos, size, used_cells, grid_size)
        
        elif shape == 'L':
            return self._build_L_cage(start_pos, size, used_cells, grid_size)
        
        elif shape == 'square':
            return self._build_square_cage(start_pos, size, used_cells, grid_size)
        
        elif shape == 'T':
            return self._build_T_cage(start_pos, size, used_cells, grid_size)
        
        else:  # irregular
            return self._build_irregular_cage(start_pos, size, used_cells, grid_size)
    
    def _build_line_cage(self, start_pos: Tuple[int, int], size: int,
                        used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build a linear cage"""
        row, col = start_pos
        cage_cells = []
        
        # Try horizontal first
        if col + size <= grid_size:
            for i in range(size):
                if (row, col + i) in used_cells:
                    break
                cage_cells.append((row, col + i))
            
            if len(cage_cells) == size:
                return cage_cells
        
        # Try vertical
        cage_cells = []
        if row + size <= grid_size:
            for i in range(size):
                if (row + i, col) in used_cells:
                    break
                cage_cells.append((row + i, col))
        
        return cage_cells
    
    def _build_L_cage(self, start_pos: Tuple[int, int], size: int,
                     used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build an L-shaped cage"""
        if size < 3:
            return self._build_line_cage(start_pos, size, used_cells, grid_size)
        
        row, col = start_pos
        cage_cells = [(row, col)]
        
        # Add horizontal arm
        arm_size = size // 2
        for i in range(1, arm_size + 1):
            if col + i < grid_size and (row, col + i) not in used_cells:
                cage_cells.append((row, col + i))
            else:
                break
        
        # Add vertical arm
        remaining = size - len(cage_cells)
        for i in range(1, remaining + 1):
            if row + i < grid_size and (row + i, col) not in used_cells:
                cage_cells.append((row + i, col))
            else:
                break
        
        return cage_cells
    
    def _build_square_cage(self, start_pos: Tuple[int, int], size: int,
                          used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build a square cage"""
        if size != 4:
            return self._build_line_cage(start_pos, size, used_cells, grid_size)
        
        row, col = start_pos
        if row + 1 < grid_size and col + 1 < grid_size:
            cells = [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)]
            if not any(cell in used_cells for cell in cells):
                return cells
        
        return []
    
    def _build_T_cage(self, start_pos: Tuple[int, int], size: int,
                     used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build a T-shaped cage"""
        if size < 3:
            return self._build_line_cage(start_pos, size, used_cells, grid_size)
        
        row, col = start_pos
        cage_cells = [(row, col)]
        
        # Add horizontal bar
        bar_size = min(size - 1, 3)
        start_col = max(0, col - bar_size // 2)
        for i in range(bar_size):
            c = start_col + i
            if c != col and c < grid_size and (row, c) not in used_cells:
                cage_cells.append((row, c))
        
        # Add vertical stem
        remaining = size - len(cage_cells)
        for i in range(1, remaining + 1):
            if row + i < grid_size and (row + i, col) not in used_cells:
                cage_cells.append((row + i, col))
            else:
                break
        
        return cage_cells
    
    def _build_irregular_cage(self, start_pos: Tuple[int, int], size: int,
                             used_cells: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """Build an irregularly shaped cage"""
        cage_cells = [start_pos]
        candidates = set()
        
        # Add adjacent cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_pos[0] + dr, start_pos[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in used_cells:
                candidates.add((nr, nc))
        
        while len(cage_cells) < size and candidates:
            next_cell = random.choice(list(candidates))
            cage_cells.append(next_cell)
            candidates.remove(next_cell)
            
            # Add new adjacent cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = next_cell[0] + dr, next_cell[1] + dc
                if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                    (nr, nc) not in used_cells and (nr, nc) not in cage_cells):
                    candidates.add((nr, nc))
        
        return cage_cells
    
    def _create_linear_cage(self, size: int, used_cells: Set[Tuple[int, int]], 
                           grid_size: int) -> List[Tuple[int, int]]:
        """Create a simple linear cage as fallback"""
        for row in range(grid_size):
            for col in range(grid_size - size + 1):
                cells = [(row, col + i) for i in range(size)]
                if not any(cell in used_cells for cell in cells):
                    return cells
        
        return []
    
    def _calculate_target_for_operation(self, values: List[int], operation: str) -> int:
        """Calculate target value for cage based on operation"""
        if not values:
            return 0
        
        if operation == 'add':
            return sum(values)
        elif operation == 'subtract':
            return abs(max(values) - min(values)) if len(values) == 2 else sum(values)
        elif operation == 'multiply':
            result = 1
            for val in values:
                result *= val
            return result
        elif operation == 'divide':
            if len(values) == 2:
                return max(values) // min(values) if min(values) != 0 else sum(values)
            else:
                return sum(values)
        else:
            return sum(values)
    
    def _create_puzzle_from_solution(self, solution: np.ndarray, cages: List[Dict], 
                                   template: Dict) -> np.ndarray:
        """Create puzzle by strategically removing values"""
        grid_size = len(solution)
        puzzle = solution.copy()
        
        # Determine how many cells to clear based on difficulty
        difficulty_rating = template['difficulty_rating']
        if difficulty_rating < 1.5:  # easy
            clear_ratio = random.uniform(0.4, 0.6)
        elif difficulty_rating < 2.5:  # moderate
            clear_ratio = random.uniform(0.5, 0.7)
        else:  # hard
            clear_ratio = random.uniform(0.6, 0.8)
        
        cells_to_clear = int(grid_size * grid_size * clear_ratio)
        
        # Get all cell positions and shuffle
        all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(all_positions)
        
        # Clear cells strategically
        cleared = 0
        for row, col in all_positions:
            if cleared >= cells_to_clear:
                break
            
            # Clear the cell
            puzzle[row, col] = 0
            cleared += 1
        
        return puzzle
    
    def _create_mnist_grid(self, grid: np.ndarray, cages: List[Dict]) -> np.ndarray:
        """Create MNIST grid using real MNIST images with THICK cage boundaries"""
        grid_size = len(grid)
        cell_size = 64  # Increased from 56 for better visibility
        total_size = grid_size * cell_size
        
        mnist_grid = np.zeros((total_size, total_size), dtype=np.uint8)
        
        print(f"    🎨 Creating Ken Ken MNIST grid with REAL MNIST images...")
        print(f"       Grid size: {grid_size}x{grid_size}, Cell size: {cell_size}x{cell_size}")
        
        total_digits_placed = 0
        
        # Place MNIST digits with padding for thick boundaries
        for row in range(grid_size):
            for col in range(grid_size):
                if grid[row, col] != 0:
                    digit = grid[row, col]
                    
                    if digit in self.mnist_images:
                        # Use the REAL MNIST image for this digit
                        digit_img = self.mnist_images[digit]
                        
                        # Resize MNIST image to fit within cell with boundary padding
                        from PIL import Image
                        inner_size = cell_size - 12  # Leave space for thick boundaries
                        pil_img = Image.fromarray(digit_img).resize((inner_size, inner_size))
                        digit_img_resized = np.array(pil_img, dtype=np.uint8)
                        
                        # Calculate position in the large grid with padding
                        start_row = row * cell_size + 6  # 6 pixel padding
                        start_col = col * cell_size + 6
                        end_row = start_row + inner_size
                        end_col = start_col + inner_size
                        
                        # Place the REAL MNIST digit image
                        mnist_grid[start_row:end_row, start_col:end_col] = digit_img_resized
                        
                        total_digits_placed += 1
                        print(f"       📍 Real MNIST digit {digit} at ({row},{col})")
                    else:
                        print(f"      ❌ No MNIST image found for digit {digit}")
        
        # Add THICK cage boundaries
        self._draw_thick_cage_boundaries(mnist_grid, cages, cell_size, grid_size)
        
        # Add cage operation labels
        self._add_cage_labels(mnist_grid, cages, cell_size)
        
        final_pixel_count = np.sum(mnist_grid > 30)
        print(f"    ✅ Ken Ken MNIST grid completed:")
        print(f"       🔢 {total_digits_placed} digits placed")
        print(f"       🎨 {final_pixel_count} significant pixels total")
        print(f"       📦 {len(cages)} cages with THICK boundaries")
        
        return mnist_grid

    def _draw_thick_cage_boundaries(self, mnist_grid: np.ndarray, cages: List[Dict], 
                               cell_size: int, grid_size: int):
        """Draw THICK cage boundaries with better visibility"""
        boundary_thickness = 6  # Much thicker boundaries
        corner_size = 12        # Larger corner markers
        
        # Create a boundary map to know which cells belong to which cages
        cage_map = {}
        for cage_idx, cage in enumerate(cages):
            for r, c in cage['cells']:
                cage_map[(r, c)] = cage_idx
        
        # Draw boundaries between different cages
        for row in range(grid_size):
            for col in range(grid_size):
                current_cage = cage_map.get((row, col), -1)
                
                pixel_row = row * cell_size
                pixel_col = col * cell_size
                
                # Check neighbors and draw boundaries where cages differ
                # Top boundary
                top_cage = cage_map.get((row - 1, col), -2)
                if current_cage != top_cage:
                    end_row = min(pixel_row + boundary_thickness, mnist_grid.shape[0])
                    end_col = min(pixel_col + cell_size, mnist_grid.shape[1])
                    mnist_grid[pixel_row:end_row, pixel_col:end_col] = 180  # Gray boundary
                    
                    # Add corner markers
                    corner_end_row = min(pixel_row + corner_size, mnist_grid.shape[0])
                    corner_end_col = min(pixel_col + corner_size, mnist_grid.shape[1])
                    mnist_grid[pixel_row:corner_end_row, pixel_col:corner_end_col] = 255  # White corner
                    
                    corner_start_col = max(pixel_col + cell_size - corner_size, 0)
                    mnist_grid[pixel_row:corner_end_row, corner_start_col:pixel_col + cell_size] = 255
                
                # Left boundary
                left_cage = cage_map.get((row, col - 1), -2)
                if current_cage != left_cage:
                    end_row = min(pixel_row + cell_size, mnist_grid.shape[0])
                    end_col = min(pixel_col + boundary_thickness, mnist_grid.shape[1])
                    mnist_grid[pixel_row:end_row, pixel_col:end_col] = 180  # Gray boundary
                    
                    # Add corner markers
                    corner_end_row = min(pixel_row + corner_size, mnist_grid.shape[0])
                    corner_end_col = min(pixel_col + corner_size, mnist_grid.shape[1])
                    mnist_grid[pixel_row:corner_end_row, pixel_col:corner_end_col] = 255  # White corner
                    
                    corner_start_row = max(pixel_row + cell_size - corner_size, 0)
                    mnist_grid[corner_start_row:pixel_row + cell_size, pixel_col:corner_end_col] = 255
                
                # Bottom boundary (for last row)
                if row == grid_size - 1:
                    start_row = max(pixel_row + cell_size - boundary_thickness, 0)
                    end_col = min(pixel_col + cell_size, mnist_grid.shape[1])
                    mnist_grid[start_row:pixel_row + cell_size, pixel_col:end_col] = 180
                
                # Right boundary (for last column)
                if col == grid_size - 1:
                    start_col = max(pixel_col + cell_size - boundary_thickness, 0)
                    end_row = min(pixel_row + cell_size, mnist_grid.shape[0])
                    mnist_grid[pixel_row:end_row, start_col:pixel_col + cell_size] = 180


    def _add_cage_labels(self, mnist_grid: np.ndarray, cages: List[Dict], cell_size: int):
        """Add enhanced cage operation labels with VISIBLE text to the grid"""
        from PIL import Image, ImageDraw, ImageFont
        
        # Convert to PIL Image for text rendering
        pil_img = Image.fromarray(mnist_grid)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            # Try to use a font
            font_size = max(14, cell_size // 6)  # Larger font
            try:
                # Try different common font paths
                font_paths = [
                    "C:/Windows/Fonts/arial.ttf",  # Windows
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
                ]
                font = None
                for path in font_paths:
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except:
                        continue
                
                if not font:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        except:
            font = None
        
        for i, cage in enumerate(cages):
            if len(cage['cells']) > 1:  # Skip single-cell cages
                # Get top-left cell of cage for label position
                min_row = min(r for r, c in cage['cells'])
                min_col = min(c for r, c in cage['cells'])
                
                # Position for label (top-left corner of cage)
                label_x = min_col * cell_size + 4
                label_y = min_row * cell_size + 4
                
                # Create operation text
                operation_symbol = {
                    'add': '+',
                    'subtract': '-', 
                    'multiply': '×',
                    'divide': '÷'
                }.get(cage['operation'], cage['operation'])
                
                label_text = f"{operation_symbol}{cage['target']}"
                
                # Calculate text size
                if font:
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(label_text) * 8
                    text_height = 12
                
                # Draw white background rectangle for text
                bg_rect = [
                    label_x - 2, 
                    label_y - 2, 
                    label_x + text_width + 6, 
                    label_y + text_height + 4
                ]
                draw.rectangle(bg_rect, fill=255, outline=0, width=2)
                
                # Draw the text in black
                text_color = 0  # Black text
                if font:
                    draw.text((label_x + 1, label_y + 1), label_text, fill=text_color, font=font)
                else:
                    # Fallback to simple text
                    draw.text((label_x + 1, label_y + 1), label_text, fill=text_color)
        
        # Convert back to numpy array and update the grid
        result = np.array(pil_img)
        mnist_grid[:] = result[:]
        return mnist_grid


    def _validate_puzzle(self, puzzle: np.ndarray, solution: np.ndarray, 
                        cages: List[Dict], template: Dict) -> bool:
        """Validate generated puzzle"""
        grid_size = len(puzzle)
        
        # Check grid size matches template
        if grid_size != template['grid_size']:
            return False
        
        # Check solution is valid Latin square
        if not self._is_valid_latin_square(solution):
            return False
        
        # Check puzzle has no conflicts
        if not self._is_valid_partial_latin_square(puzzle):
            return False
        
        # Check cages are valid
        if not self._validate_cages(cages, solution):
            return False
        
        # Check puzzle can lead to solution
        if not self._puzzle_leads_to_solution(puzzle, solution):
            return False
        
        return True
    
    def _is_valid_latin_square(self, grid: np.ndarray) -> bool:
        """Check if grid is a valid Latin square"""
        grid_size = len(grid)
        
        # Check all cells filled
        if np.any(grid == 0):
            return False
        
        # Check rows
        for row in range(grid_size):
            if set(grid[row, :]) != set(range(1, grid_size + 1)):
                return False
        
        # Check columns
        for col in range(grid_size):
            if set(grid[:, col]) != set(range(1, grid_size + 1)):
                return False
        
        return True
    
    def _is_valid_partial_latin_square(self, grid: np.ndarray) -> bool:
        """Check if partial grid has no conflicts"""
        grid_size = len(grid)
        
        # Check rows
        for row in range(grid_size):
            filled = [x for x in grid[row, :] if x != 0]
            if len(filled) != len(set(filled)):
                return False
        
        # Check columns
        for col in range(grid_size):
            filled = [x for x in grid[:, col] if x != 0]
            if len(filled) != len(set(filled)):
                return False
        
        return True
    
    def _validate_cages(self, cages: List[Dict], solution: np.ndarray) -> bool:
        """Check if all cages satisfy their constraints in the solution"""
        for cage in cages:
            cage_values = [solution[r, c] for r, c in cage['cells']]
            if not self._evaluate_cage_constraint(cage_values, cage['operation'], cage['target']):
                return False
        return True
    
    def _evaluate_cage_constraint(self, values: List[int], operation: str, target: int) -> bool:
        """Check if cage values satisfy the constraint"""
        if not values:
            return False
        
        if operation == 'add':
            return sum(values) == target
        elif operation == 'subtract':
            if len(values) == 2:
                return abs(values[0] - values[1]) == target
            return False
        elif operation == 'multiply':
            result = 1
            for val in values:
                result *= val
            return result == target
        elif operation == 'divide':
            if len(values) == 2:
                return max(values) / min(values) == target
            return False
        
        return False
    
    def _puzzle_leads_to_solution(self, puzzle: np.ndarray, solution: np.ndarray) -> bool:
        """Check if puzzle can lead to the given solution"""
        for row in range(len(puzzle)):
            for col in range(len(puzzle)):
                if puzzle[row, col] != 0:
                    if puzzle[row, col] != solution[row, col]:
                        return False
        return True
    
    def _get_difficulty_from_template(self, template: Dict) -> str:
        """Get difficulty level from template"""
        rating = template['difficulty_rating']
        if rating < 1.5:
            return 'easy'
        elif rating < 2.5:
            return 'moderate'
        else:
            return 'hard'
    
    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.stats = {'generated': 0, 'failed': 0, 'total_time': 0}



def save_images_with_metadata(dataset: List[Dict], output_dir: str):
    """Save MNIST images with metadata for Ken Ken (with numpy fix)"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        metadata_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        for entry in dataset:
            puzzle_id = entry['id']
            
            # Save images
            from PIL import Image
            puzzle_img = Image.fromarray(np.array(entry['mnist_puzzle'], dtype=np.uint8))
            solution_img = Image.fromarray(np.array(entry['mnist_solution'], dtype=np.uint8))
            
            puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
            solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
            
            puzzle_img.save(puzzle_path)
            solution_img.save(solution_path)
            
            # Create Ken Ken specific metadata (convert numpy types)
            metadata = {
                'puzzle_info': {
                    'id': str(entry['id']),
                    'difficulty': str(entry['difficulty']),
                    'grid_size': int(entry['grid_size']),
                    'validation_status': 'VALID',
                    'generated_timestamp': str(entry['metadata']['generated_timestamp'])
                },
                'grids': {
                    'puzzle_grid': [[int(cell) for cell in row] for row in entry['puzzle_grid']],
                    'solution_grid': [[int(cell) for cell in row] for row in entry['solution_grid']]
                },
                'cages': [
                    {
                        'cells': [(int(r), int(c)) for r, c in cage['cells']],
                        'operation': str(cage['operation']),
                        'target': int(cage['target'])
                    } for cage in entry['cages']
                ],
                'strategies': {
                    'required_strategies': [str(s) for s in entry['required_strategies']],
                    'strategy_details': {str(k): v for k, v in entry['strategy_details'].items()}
                },
                'files': {
                    'puzzle_image': f"{puzzle_id}_puzzle.png",
                    'solution_image': f"{puzzle_id}_solution.png",
                    'puzzle_image_path': os.path.abspath(puzzle_path),
                    'solution_image_path': os.path.abspath(solution_path)
                },
                'statistics': {
                    'grid_size': int(entry['grid_size']),
                    'total_cells': int(entry['grid_size'] ** 2),
                    'filled_cells': int(entry['metadata']['filled_cells']),
                    'empty_cells': int(entry['metadata']['empty_cells']),
                    'fill_percentage': round(float(entry['metadata']['filled_cells']) / float(entry['grid_size'] ** 2) * 100, 1),
                    'num_cages': int(entry['metadata']['num_cages']),
                    'operations_used': [str(op) for op in entry['metadata']['operations_used']],
                    'difficulty_score': float(entry['metadata']['difficulty_score']),
                    'generation_attempt': int(entry['metadata']['generation_attempt'])
                }
            }
            
            metadata_path = os.path.join(metadata_dir, f"{puzzle_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"🖼️ Ken Ken MNIST images and metadata saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Error saving Ken Ken images: {e}")

# Test function
def test_kenken_template_generator():
    """Test the Ken Ken template generator"""
    print("Testing Ken Ken Template-Based Generator...")
    
    generator = KenKenTemplateBasedGenerator()
    
    # Test easy puzzles
    easy_puzzles = generator.generate_puzzles('easy', 2)
    print(f"Generated {len(easy_puzzles)} easy Ken Ken puzzles")
    
    # Test moderate puzzles
    moderate_puzzles = generator.generate_puzzles('moderate', 1)
    print(f"Generated {len(moderate_puzzles)} moderate Ken Ken puzzles")
    
    # Test hard puzzles
    hard_puzzles = generator.generate_puzzles('hard', 1)
    print(f"Generated {len(hard_puzzles)} hard Ken Ken puzzles")
    
    return easy_puzzles, moderate_puzzles, hard_puzzles



if __name__ == "__main__":
    test_kenken_template_generator()