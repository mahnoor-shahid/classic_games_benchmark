# enhanced_image_generator.py
"""
Enhanced Image Generator for Futoshiki Puzzles
Creates puzzle images with visible constraint symbols (< and >) directly on the images
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Tuple

class EnhancedFutoshikiImageGenerator:
    """Enhanced image generator that adds constraint symbols directly to puzzle images"""
    
    def __init__(self):
        self.cell_size = 60  # Larger cells for better visibility
        self.constraint_font_size = 20
        self.digit_font_size = 36
        
    def create_puzzle_image_with_constraints(self, mnist_grid: np.ndarray, 
                                           h_constraints: Dict, v_constraints: Dict, 
                                           size: int, puzzle_type: str = "puzzle") -> Image.Image:
        """Create a puzzle image with constraint symbols overlaid"""
        
        # Calculate dimensions
        grid_size = size * self.cell_size
        margin = 40  # Extra margin for constraint symbols
        img_width = grid_size + 2 * margin
        img_height = grid_size + 2 * margin
        
        # Create white background image
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            digit_font = ImageFont.truetype("arial.ttf", self.digit_font_size)
            constraint_font = ImageFont.truetype("arial.ttf", self.constraint_font_size)
        except:
            try:
                digit_font = ImageFont.load_default()
                constraint_font = ImageFont.load_default()
            except:
                digit_font = None
                constraint_font = None
        
        # Draw the MNIST grid as background (resized to fit our cell size)
        if mnist_grid is not None and mnist_grid.size > 0:
            mnist_img = Image.fromarray(mnist_grid, mode='L')
            # Resize MNIST to fit our grid
            mnist_resized = mnist_img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
            # Paste MNIST background
            img.paste(mnist_resized, (margin, margin))
        
        # Draw grid lines
        for i in range(size + 1):
            # Vertical lines
            x = margin + i * self.cell_size
            draw.line([(x, margin), (x, margin + grid_size)], fill='black', width=3)
            
            # Horizontal lines
            y = margin + i * self.cell_size
            draw.line([(margin, y), (margin + grid_size, y)], fill='black', width=3)
        
        # Parse constraint dictionaries if they're string-keyed
        parsed_h_constraints = {}
        parsed_v_constraints = {}
        
        for key, value in h_constraints.items():
            if isinstance(key, str):
                row, col = map(int, key.split(','))
                parsed_h_constraints[(row, col)] = value
            else:
                parsed_h_constraints[key] = value
        
        for key, value in v_constraints.items():
            if isinstance(key, str):
                row, col = map(int, key.split(','))
                parsed_v_constraints[(row, col)] = value
            else:
                parsed_v_constraints[key] = value
        
        # Draw horizontal constraint symbols
        for (row, col), constraint in parsed_h_constraints.items():
            if col < size - 1:  # Make sure we don't go out of bounds
                # Position between cells
                x = margin + (col + 0.5) * self.cell_size + self.cell_size // 2
                y = margin + row * self.cell_size + self.cell_size // 2
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw white background circle for symbol
                circle_radius = 15
                draw.ellipse([x - circle_radius, y - circle_radius, 
                             x + circle_radius, y + circle_radius], 
                            fill='yellow', outline='red', width=2)
                
                # Draw the constraint symbol
                if constraint_font:
                    draw.text((x, y), symbol, fill='red', font=constraint_font, anchor='mm')
                else:
                    draw.text((x, y), symbol, fill='red', anchor='mm')
        
        # Draw vertical constraint symbols
        for (row, col), constraint in parsed_v_constraints.items():
            if row < size - 1:  # Make sure we don't go out of bounds
                # Position between cells
                x = margin + col * self.cell_size + self.cell_size // 2
                y = margin + (row + 0.5) * self.cell_size + self.cell_size // 2
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw white background circle for symbol
                circle_radius = 15
                draw.ellipse([x - circle_radius, y - circle_radius, 
                             x + circle_radius, y + circle_radius], 
                            fill='lightblue', outline='blue', width=2)
                
                # Draw the constraint symbol
                if constraint_font:
                    draw.text((x, y), symbol, fill='blue', font=constraint_font, anchor='mm')
                else:
                    draw.text((x, y), symbol, fill='blue', anchor='mm')
        
        # Add title
        title = f"Futoshiki {puzzle_type.capitalize()} ({size}x{size})"
        if constraint_font:
            draw.text((img_width // 2, 15), title, fill='black', font=constraint_font, anchor='mt')
        else:
            draw.text((img_width // 2, 15), title, fill='black', anchor='mt')
        
        # Add constraint legend
        legend_y = img_height - 25
        legend_text = f"Constraints: {len(parsed_h_constraints)} horizontal (yellow), {len(parsed_v_constraints)} vertical (blue)"
        if constraint_font:
            draw.text((img_width // 2, legend_y), legend_text, fill='gray', font=constraint_font, anchor='mb')
        else:
            draw.text((img_width // 2, legend_y), legend_text, fill='gray', anchor='mb')
        
        return img
    
    def create_enhanced_puzzle_images(self, puzzle_data: Dict, output_dir: str):
        """Create enhanced puzzle and solution images with visible constraints"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            puzzle_id = puzzle_data['id']
            size = puzzle_data['size']
            
            # Get MNIST grids
            mnist_puzzle = np.array(puzzle_data['mnist_puzzle'], dtype=np.uint8)
            mnist_solution = np.array(puzzle_data['mnist_solution'], dtype=np.uint8)
            
            # Get constraints
            h_constraints = puzzle_data['h_constraints']
            v_constraints = puzzle_data['v_constraints']
            
            # Create enhanced puzzle image
            puzzle_img = self.create_puzzle_image_with_constraints(
                mnist_puzzle, h_constraints, v_constraints, size, "puzzle"
            )
            
            # Create enhanced solution image
            solution_img = self.create_puzzle_image_with_constraints(
                mnist_solution, h_constraints, v_constraints, size, "solution"
            )
            
            # Save images
            puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle.png")
            solution_path = os.path.join(output_dir, f"{puzzle_id}_solution.png")
            
            puzzle_img.save(puzzle_path, quality=95, optimize=True)
            solution_img.save(solution_path, quality=95, optimize=True)
            
            print(f"    🖼️ Enhanced images with constraints saved: {puzzle_id}")
            return True
            
        except Exception as e:
            print(f"    ⚠️ Error creating enhanced images for {puzzle_data.get('id', 'unknown')}: {e}")
            return False
    
    def create_grid_only_image(self, puzzle_grid: List[List[int]], solution_grid: List[List[int]], 
                              h_constraints: Dict, v_constraints: Dict, size: int, puzzle_id: str, output_dir: str):
        """Create clean grid images without MNIST background"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create puzzle grid image
            puzzle_img = self._create_clean_grid_image(puzzle_grid, h_constraints, v_constraints, size, "puzzle")
            
            # Create solution grid image
            solution_img = self._create_clean_grid_image(solution_grid, h_constraints, v_constraints, size, "solution")
            
            # Save images
            puzzle_path = os.path.join(output_dir, f"{puzzle_id}_puzzle_clean.png")
            solution_path = os.path.join(output_dir, f"{puzzle_id}_solution_clean.png")
            
            puzzle_img.save(puzzle_path, quality=95, optimize=True)
            solution_img.save(solution_path, quality=95, optimize=True)
            
            print(f"    📋 Clean grid images saved: {puzzle_id}")
            return True
            
        except Exception as e:
            print(f"    ⚠️ Error creating clean grid images for {puzzle_id}: {e}")
            return False
    
    def _create_clean_grid_image(self, grid: List[List[int]], h_constraints: Dict, v_constraints: Dict, 
                                size: int, puzzle_type: str) -> Image.Image:
        """Create a clean grid image with numbers and constraint symbols"""
        
        # Calculate dimensions
        grid_size = size * self.cell_size
        margin = 40
        img_width = grid_size + 2 * margin
        img_height = grid_size + 2 * margin
        
        # Create white background image
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            digit_font = ImageFont.truetype("arial.ttf", self.digit_font_size)
            constraint_font = ImageFont.truetype("arial.ttf", self.constraint_font_size)
        except:
            digit_font = ImageFont.load_default()
            constraint_font = ImageFont.load_default()
        
        # Draw grid lines
        for i in range(size + 1):
            # Vertical lines
            x = margin + i * self.cell_size
            draw.line([(x, margin), (x, margin + grid_size)], fill='black', width=2)
            
            # Horizontal lines
            y = margin + i * self.cell_size
            draw.line([(margin, y), (margin + grid_size, y)], fill='black', width=2)
        
        # Draw numbers
        for row in range(size):
            for col in range(size):
                if grid[row][col] != 0:
                    x = margin + col * self.cell_size + self.cell_size // 2
                    y = margin + row * self.cell_size + self.cell_size // 2
                    
                    number = str(grid[row][col])
                    color = 'blue' if puzzle_type == "puzzle" else 'green'
                    
                    draw.text((x, y), number, fill=color, font=digit_font, anchor='mm')
        
        # Parse constraint dictionaries
        parsed_h_constraints = {}
        parsed_v_constraints = {}
        
        for key, value in h_constraints.items():
            if isinstance(key, str):
                row, col = map(int, key.split(','))
                parsed_h_constraints[(row, col)] = value
            else:
                parsed_h_constraints[key] = value
        
        for key, value in v_constraints.items():
            if isinstance(key, str):
                row, col = map(int, key.split(','))
                parsed_v_constraints[(row, col)] = value
            else:
                parsed_v_constraints[key] = value
        
        # Draw horizontal constraint symbols
        for (row, col), constraint in parsed_h_constraints.items():
            if col < size - 1:
                x = margin + (col + 0.5) * self.cell_size + self.cell_size // 2
                y = margin + row * self.cell_size + self.cell_size // 2
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw background
                circle_radius = 12
                draw.ellipse([x - circle_radius, y - circle_radius, 
                             x + circle_radius, y + circle_radius], 
                            fill='yellow', outline='red', width=2)
                
                draw.text((x, y), symbol, fill='red', font=constraint_font, anchor='mm')
        
        # Draw vertical constraint symbols
        for (row, col), constraint in parsed_v_constraints.items():
            if row < size - 1:
                x = margin + col * self.cell_size + self.cell_size // 2
                y = margin + (row + 0.5) * self.cell_size + self.cell_size // 2
                
                symbol = '<' if constraint == '<' else '>'
                
                # Draw background
                circle_radius = 12
                draw.ellipse([x - circle_radius, y - circle_radius, 
                             x + circle_radius, y + circle_radius], 
                            fill='lightblue', outline='blue', width=2)
                
                draw.text((x, y), symbol, fill='blue', font=constraint_font, anchor='mm')
        
        # Add title
        title = f"Futoshiki {puzzle_type.capitalize()} ({size}x{size})"
        draw.text((img_width // 2, 15), title, fill='black', font=constraint_font, anchor='mt')
        
        return img


# Updated FreshTemplateFutoshikiGenerator with enhanced image generation
def add_enhanced_image_methods():
    """
    This function contains the updated methods to add to FreshTemplateFutoshikiGenerator
    Copy these methods into the FreshTemplateFutoshikiGenerator class
    """
    
    def save_enhanced_images_with_constraints(self, dataset: List[Dict], output_dir: str):
        """Save enhanced images with visible constraint symbols"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize enhanced image generator
            image_generator = EnhancedFutoshikiImageGenerator()
            
            for puzzle_data in dataset:
                # Create enhanced images with constraints
                image_generator.create_enhanced_puzzle_images(puzzle_data, output_dir)
                
                # Also create clean grid versions
                image_generator.create_grid_only_image(
                    puzzle_data['puzzle_grid'],
                    puzzle_data['solution_grid'], 
                    puzzle_data['h_constraints'],
                    puzzle_data['v_constraints'],
                    puzzle_data['size'],
                    puzzle_data['id'],
                    output_dir
                )
            
            print(f"🖼️ Enhanced images with visible constraints saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving enhanced images: {e}")
    
    def create_single_puzzle_image_with_constraints(self, puzzle_data: Dict, output_dir: str):
        """Create a single puzzle image with visible constraints for testing"""
        image_generator = EnhancedFutoshikiImageGenerator()
        return image_generator.create_enhanced_puzzle_images(puzzle_data, output_dir)


# Example usage and test function
def test_enhanced_image_generation():
    """Test the enhanced image generation"""
    print("🧪 Testing Enhanced Image Generation with Constraints")
    print("=" * 60)
    
    # Import the generator
    try:
        from futoshiki_template_generator import FreshTemplateFutoshikiGenerator
        
        generator = FreshTemplateFutoshikiGenerator()
        
        # Generate a test puzzle
        puzzle_data = generator.generate_puzzle_from_template('easy', 0)
        puzzle_id = "test_enhanced_001"
        
        # Create MNIST representations
        mnist_puzzle = generator.create_mnist_representation(puzzle_data['puzzle'], puzzle_id)
        mnist_solution = generator.create_mnist_representation(puzzle_data['solution'], puzzle_id)
        
        # Create constraint visualization
        constraint_viz = generator.create_enhanced_constraint_visualization(
            puzzle_data['h_constraints'], puzzle_data['v_constraints'], 5
        )
        
        # Create test puzzle entry
        test_puzzle = {
            'id': puzzle_id,
            'size': 5,
            'puzzle_grid': puzzle_data['puzzle'].tolist(),
            'solution_grid': puzzle_data['solution'].tolist(),
            'h_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['h_constraints'].items()},
            'v_constraints': {f"{k[0]},{k[1]}": v for k, v in puzzle_data['v_constraints'].items()},
            'mnist_puzzle': mnist_puzzle.tolist(),
            'mnist_solution': mnist_solution.tolist(),
            'constraint_visualization': constraint_viz
        }
        
        # Create enhanced images
        image_generator = EnhancedFutoshikiImageGenerator()
        output_dir = "./test_enhanced_images"
        
        success = image_generator.create_enhanced_puzzle_images(test_puzzle, output_dir)
        
        if success:
            print("✅ Enhanced images with constraints created successfully!")
            print(f"📁 Check the '{output_dir}' folder for:")
            print(f"   • {puzzle_id}_puzzle.png (with constraint symbols)")
            print(f"   • {puzzle_id}_solution.png (with constraint symbols)")
        else:
            print("❌ Failed to create enhanced images")
        
        # Also create clean grid versions
        image_generator.create_grid_only_image(
            test_puzzle['puzzle_grid'],
            test_puzzle['solution_grid'],
            test_puzzle['h_constraints'],
            test_puzzle['v_constraints'],
            5,
            puzzle_id,
            output_dir
        )
        
        print(f"📋 Clean grid images also created:")
        print(f"   • {puzzle_id}_puzzle_clean.png")
        print(f"   • {puzzle_id}_solution_clean.png")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_image_generation()