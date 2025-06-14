# MNIST Sudoku Project

A comprehensive Sudoku puzzle generator using MNIST dataset with compositional strategy knowledge bases for testing AI compositionality.

## Overview

This project generates Sudoku puzzles of varying difficulties using MNIST digit images and employs First-Order Logic (FOL) rules to define solving strategies. The key innovation is the compositional structure where:

- **Easy strategies** are atomic/base-level solving techniques
- **Moderate strategies** compose easy strategies
- **Hard strategies** compose both easy and moderate strategies

This hierarchical compositionality makes it ideal for testing AI systems' ability to understand and apply compositional reasoning.

## Project Structure

```
mnist-sudoku-project/
├── easy_strategies_kb.py          # Easy strategies knowledge base
├── moderate_strategies_kb.py      # Moderate strategies knowledge base  
├── hard_strategies_kb.py          # Hard strategies knowledge base
├── sudoku_generator.py            # Main puzzle generator using MNIST
├── puzzle_solver.py               # Sudoku solver implementing strategies
├── dataset_analyzer.py            # Dataset analysis and validation
├── main.py                        # Main orchestration script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

### Knowledge Bases
- **10 Easy Strategies**: Naked singles, hidden singles, full house, candidate elimination
- **9 Moderate Strategies**: Naked/hidden pairs/triples, pointing pairs, X-Wing, XY-Wing
- **14 Hard Strategies**: Swordfish, ALS chains, coloring, exotic patterns like Exocet

### Puzzle Generation
- Generates puzzles solvable with specific strategy combinations
- Uses real MNIST digit images (28x28) for visual representation
- Creates 252x252 pixel puzzle images (9x9 grid of MNIST digits)
- Ensures unique solutions and appropriate difficulty levels

### Dataset Output
Each generated puzzle includes:
- **x**: Puzzle grid with MNIST digit images
- **y**: Required solving strategies (compositional)
- **solution**: Complete MNIST solution image
- Strategy metadata and FOL rules

## Installation & Setup

### 🚀 Quick Setup (Recommended)

**Linux/Mac:**
```bash
# Make setup script executable and run
chmod +x quick_setup.sh
./quick_setup.sh
```

**Windows:**
```cmd
# Run the setup batch file
quick_setup.bat
```

### 📋 Manual Setup

```bash
# Clone the repository
git clone <repository-url>
cd mnist-sudoku-project

# Install dependencies
pip install -r requirements.txt

# Run setup with configuration
python setup.py

# Or run with custom config
python setup.py --config my_config.yaml
```

### ⚙️ Configuration

The project uses a comprehensive `config.yaml` file for all settings:

```yaml
# Example configuration
project:
  name: "mnist-sudoku"
  output_dir: "./output"

generation:
  num_puzzles:
    easy: 100
    moderate: 75
    hard: 50
  
  complexity:
    easy:
      min_filled_cells: 35
      max_filled_cells: 45
      max_strategies: 2

hardware:
  device: "auto"  # auto, cpu, cuda, mps
  num_workers: 4
  memory_limit_gb: 8

analysis:
  generate_plots: true
  plot_formats: ["png", "pdf"]
  
output:
  formats:
    json: true
    csv: true
  images:
    save_mnist_puzzles: true
    compress_images: true
```

## Usage

### Quick Start - Full Pipeline
```bash
python main.py --action full --num_puzzles 50
```

### Individual Components

#### 1. View Knowledge Bases
```bash
python main.py --action show_kb
```

#### 2. Generate Datasets
```bash
python main.py --action generate --num_puzzles 100
```

#### 3. Analyze Datasets
```bash
python main.py --action analyze
```

#### 4. Test Solver
```bash
python main.py --action test
```

#### 5. Create Sample Puzzle
```bash
python main.py --action sample --difficulty moderate
```

## Knowledge Base Structure

### Easy Strategies (Base Level)
1. **Naked Single**: Cell has only one possible value
2. **Hidden Single (Row/Column/Box)**: Value can only go in one cell
3. **Eliminate Candidates**: Remove assigned values from peers
4. **Full House**: Fill last empty cell in unit

### Moderate Strategies (Compositional Level 1)
1. **Naked Pair/Triple**: Cells with identical candidate sets
2. **Hidden Pair/Triple**: Values confined to specific cells
3. **Pointing Pairs**: Box-line interactions
4. **Box/Line Reduction**: Line-box interactions
5. **XY-Wing**: Three-cell elimination pattern
6. **Simple Coloring**: Conjugate pair chains
7. **X-Wing**: Four-corner elimination pattern

### Hard Strategies (Compositional Level 2+)
1. **Swordfish/Jellyfish**: Extended fish patterns
2. **XYZ-Wing/WXYZ-Wing**: Advanced wing patterns
3. **ALS-XZ Rule**: Almost Locked Set interactions
4. **Sue de Coq**: Complex box-line patterns
5. **Death Blossom**: Multi-ALS stem patterns
6. **Multi-Coloring**: Multiple chain interactions
7. **SK Loop**: Strong link loops
8. **AIC**: Alternating Inference Chains
9. **Exocet/Junior Exocet**: Exotic elimination patterns

## FOL Rule Examples

### Easy Strategy Example
```
Naked Single:
∀cell(r,c) ∀value(v): 
    [candidates(cell(r,c)) = {v}] → [assign(cell(r,c), v)]
```

### Moderate Strategy Example (Compositional)
```
Naked Pair:
∀unit(u) ∀cell(r1,c1) ∀cell(r2,c2) ∀value_set{v1,v2}:
    [cell(r1,c1) ∈ unit(u)] ∧ [cell(r2,c2) ∈ unit(u)] ∧ [(r1,c1) ≠ (r2,c2)]
    ∧ [candidates(cell(r1,c1)) = {v1,v2}] ∧ [candidates(cell(r2,c2)) = {v1,v2}]
    → [∀cell(r',c') ∈ unit(u), (r',c') ∉ {(r1,c1),(r2,c2)}: 
       remove_candidate(cell(r',c'), v1) ∧ remove_candidate(cell(r',c'), v2)]

Composed of: [eliminate_candidates_row, eliminate_candidates_column, eliminate_candidates_box]
```

## Output Files

### Generated Datasets
- `sudoku_dataset_easy.json`: Easy puzzles with metadata
- `sudoku_dataset_moderate.json`: Moderate puzzles with metadata
- `sudoku_dataset_hard.json`: Hard puzzles with metadata

### MNIST Images
- `mnist_sudoku_images_easy/`: PNG files for easy puzzles
- `mnist_sudoku_images_moderate/`: PNG files for moderate puzzles  
- `mnist_sudoku_images_hard/`: PNG files for hard puzzles

### Analysis Files
- `analysis_report.txt`: Comprehensive dataset analysis
- `strategy_composition_graph.json`: Strategy dependency graph
- `analysis_plots_*/`: Visualization plots for each difficulty

## Dataset Schema

```json
{
  "id": "easy_0001",
  "difficulty": "easy",
  "puzzle_grid": [[0,5,0,...], ...],
  "solution_grid": [[4,5,6,...], ...],
  "required_strategies": ["naked_single", "hidden_single_row"],
  "mnist_puzzle": [[0,0,255,...], ...],
  "mnist_solution": [[0,0,255,...], ...],
  "strategy_details": {
    "naked_single": {
      "name": "Naked Single",
      "description": "A cell has only one possible value",
      "fol_rule": "∀cell(r,c) ∀value(v): ...",
      "logic": "If a cell has only one candidate value, assign that value",
      "complexity": "easy",
      "composite": false
    }
  }
}
```

## Compositionality Testing

This dataset is specifically designed for testing AI compositionality:

1. **Hierarchical Structure**: Easy → Moderate → Hard
2. **Explicit Composition**: Each strategy lists its components
3. **FOL Representation**: Formal logic for precise understanding
4. **Visual + Logical**: MNIST images + strategy requirements
5. **Validation**: Ensures puzzles are solvable with specified strategies

## Research Applications

- **Compositional Reasoning**: Test AI understanding of strategy composition
- **Transfer Learning**: Learn easy strategies, apply to moderate/hard
- **Multi-modal Learning**: Visual (MNIST) + symbolic (strategies)
- **Logical Reasoning**: FOL rule application and inference
- **Puzzle Solving**: Benchmark for constraint satisfaction

## Validation Features

- **Solution Correctness**: Verifies all solutions are valid
- **Strategy Sufficiency**: Ensures puzzles solvable with required strategies
- **Compositionality Check**: Validates strategy dependency structure
- **Difficulty Progression**: Confirms appropriate complexity scaling

## Customization

### Adding New Strategies
1. Add strategy to appropriate knowledge base file
2. Implement in `puzzle_solver.py`
3. Define composition relationships
4. Update FOL rules

### Adjusting Difficulty
- Modify removal patterns in `sudoku_generator.py`
- Adjust strategy selection logic
- Customize validation criteria

## Performance

- **Generation Speed**: ~1-2 puzzles/second
- **Dataset Size**: 50-100 puzzles per difficulty recommended
- **Memory Usage**: ~100MB for 300 puzzles with MNIST images
- **Validation**: 95%+ puzzle validity rate

## Known Limitations

1. **Strategy Implementation**: Some advanced strategies are simplified
1. **MNIST Dependency**: Requires PyTorch for MNIST loading
3. **Solving Completeness**: Solver may not find all possible strategy applications
4. **Generation Time**: Complex puzzles take longer to generate

## Contributing

1. Fork the repository
2. Add new strategies to knowledge bases
3. Implement solver methods
4. Add tests and validation
5. Submit pull request

## License

[Add your license information here]

## Citation

If you use this dataset in research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]