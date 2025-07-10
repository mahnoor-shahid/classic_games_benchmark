# Classic Games Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-purple)](https://matplotlib.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.18%2B-red)](https://scikit-image.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-red)](https://scikit-learn.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.62%2B-pink)](https://tqdm.github.io/)
[![PyYAML](https://img.shields.io/badge/PyYAML-5.4%2B-orange)](https://pyyaml.org/)

> **A collection of classic logic puzzle games implemented with basic python for evaluating compositional generalization.**

This project is developed by the [Department of Sustainability and Innovation in Digital Ecosystems](https://www.sust.wiwi.uni-due.de) at the University of Duisburg-Essen.

### 🤖 **AI-Powered Puzzle Suite with Compositional Reasoning**
- **Template-Based**: Lightning-fast puzzle creation
- **Strategy-Driven**: Puzzles require specific solving techniques + compositionality
- **Difficulty Scaling**: Mathematically validated difficulty progression
- **Uniqueness Guaranteed**: Every puzzle has exactly one solution
---
## 🎲 Puzzle Games

<table>
<tr>
<td width="33%" align="center">

### 🔢 **Sudoku**
**MNIST-Enhanced Classic**

- 🖼️ **Visual Recognition**: Real MNIST digit input
- 🧩 **Smart Generation**: Strategy-based puzzle creation
- 📊 **Difficulty Scaling**: Easy to Expert levels
- 🔍 **Strategy Hierarchy**: 15+ solving techniques

[**Play Sudoku →**](game_projects/sudoku/)

</td>
<td width="33%" align="center">

### ➕ **KenKen**
**Arithmetic Puzzle Master**

- 🔢 **Operation Recognition**: Visual math operator detection
- 📐 **Variable Grid Sizes**: 3x3 to 9x9 support
- 🎯 **Strategy Hierarchy**: 10+ solving techniques
- 📈 🔗 **Compositionality**: Validated strategy chains

[**Play KenKen →**](game_projects/kenken/)

</td>
<td width="33%" align="center">

### ➕ **Kakuro**
**Cross-Sum Logic Puzzle**

- 🧮 **Sum Constraints**: Row and column clues
- 🖼️ **MNIST Digits**: Realistic number representation
- 🧩 **Template-Based Generation**: Multiple symmetry types
- 📊 **Difficulty Scaling**: Easy, Moderate, Hard
- 🧠 **Strategy Hierarchy**: 10+ solving techniques
- 🔗 **Compositionality**: Validated strategy chains

[**Play Kakuro →**](game_projects/kakuro/)

</td>
</tr>
<tr>
<td width="33%" align="center">

### ⚖️ **Futoshiki**
**Inequality Logic Challenge**

- 🔤 **Symbol Recognition**: Automatic < > detection
- 🎨 **Visual Constraints**: MNIST digits + constraint symbols
- 🧠 **Strategy Hierarchy**: 25+ solving techniques
- 🔗 **Compositionality**: Validated strategy chains

[**Play Futoshiki →**](game_projects/futoshiki/)

</td>
</tr>
</table>

---

## 🚀 Getting Started

### ⚡ **One-Command Setup**
Get everything running in under 2 minutes:

```bash
# Clone and setup everything automatically
git clone https://github.com/yourusername/classic-games.git
cd classic-games
python unified_env_quick_setup.py
```

This magical script will:
- ✅ Create isolated virtual environment
- ✅ Install all dependencies automatically
- ✅ Download MNIST 
- ✅ Validate installation with test puzzles

### 🎯 Manual Setup
If you prefer to set up manually:

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

#### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/classic-games.git
cd classic-games
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 🎮 **Start Playing Immediately**

```bash
# Generate and solve Sudoku puzzles
cd game_projects/sudoku
python main.py --action generate_validated 

# Create KenKen challenges
cd game_projects/kenken  
python main.py --action generate_validated 

# Build Futoshiki with visual constraints
cd game_projects/futoshiki
python futoshiki_main.py --action generate 

# Generate Kakuro cross-sum puzzles
cd game_projects/kakuro
python main.py --action generate_validated --difficulty easy --count 10
```

---

## 📂 **Organized Project Structure**

```
classic-games/
├── 🎮 game_projects/
│   ├── 🧩 sudoku/                    # Complete Sudoku implementation
│   │   ├── main.py                   # Main game interface
│   │   ├── generator.py              # Puzzle generation engine
│   │   ├── solver.py                 # AI solving strategies
│   │   ├── knowledge_base/           # Strategy definitions
│   │   └── templates/                # Puzzle templates
│   ├── ➕ kenken/                    # KenKen arithmetic puzzles
│   │   ├── main.py
│   │   ├── arithmetic_engine.py
│   │   ├── constraint_solver.py
│   │   └── visual_recognition.py
│   ├── ➕ kakuro/                    # Kakuro cross-sum puzzles
│   │   ├── main.py                   # Main game interface
│   │   ├── template_based_generators.py # Puzzle generation engine
│   │   ├── kakuro_validator.py       # Puzzle validation logic
│   │   ├── kakuro_easy_strategies_kb.py # Easy strategies knowledge base
│   │   ├── kakuro_moderate_strategies_kb.py # Moderate strategies knowledge base
│   │   ├── kakuro_hard_strategies_kb.py # Hard strategies knowledge base
│   │   └── config.yaml               # Configuration
│   └── ⚖️ futoshiki/                 # Inequality constraint puzzles
│       ├── futoshiki_main.py
│       ├── template_generator.py
│       ├── constraint_visualizer.py
│       └── strategy_validator.py
├── 📋 requirements.txt               # Python dependencies
├── 🔧 unified_env_quick_setup.py    # One-command setup
├── 📊 benchmarks/                    # Performance benchmarks
├── 📖 docs/                          # Documentation
└── 🧪 tests/                         # Comprehensive tests
```

### 🧠 **Knowledge Base System**

Each game implements a hierarchical knowledge base:

```python
# Example: Sudoku Strategy Hierarchy
Easy Strategies (Base Level)
├── naked_single
├── constraint_propagation  
├── row_uniqueness
└── column_uniqueness

Moderate Strategies (Composed)
├── naked_pair → [naked_single + constraint_propagation]
├── hidden_pair → [hidden_single + uniqueness]
└── constraint_chains → [propagation + forcing]

Hard Strategies (Advanced Composition)
├── multiple_chains → [chain_analysis + intersection]
├── network_analysis → [global_consistency + propagation]
└── temporal_reasoning → [sequence_analysis + validation]

# Example: Kakuro Strategy Hierarchy
Easy Strategies (Base Level)
├── single_cell_sum
├── unique_sum_combination
├── cross_reference
├── eliminate_impossible
├── sum_partition
├── digit_frequency

Moderate Strategies (Composed)
├── sum_partition → [single_cell_sum + unique_sum_combination]
├── digit_frequency → [cross_reference + eliminate_impossible]
├── sum_difference → [sum_partition + digit_frequency]
├── minimum_maximum → [sum_partition + sum_difference]
├── sum_completion → [sum_partition + minimum_maximum]
├── digit_elimination → [digit_frequency + sum_completion]

Hard Strategies (Advanced Composition)
├── sum_completion → [sum_partition + minimum_maximum + sum_difference]
├── digit_elimination → [digit_frequency + sum_completion + cross_reference]
├── sum_difference → [sum_partition + digit_frequency + cross_reference]
├── minimum_maximum → [sum_partition + sum_difference + unique_sum_combination]
```

---


## 🤝 Contributing

We welcome contributions to expand our collection of classic games and puzzles! This project is specifically designed to test and demonstrate compositionality in puzzle-solving strategies and knowledge bases. If you're interested in contributing, here's how you can help:

### Adding New Games
We're particularly interested in games that:
- Have clear, composable solving strategies
- Can be represented with a knowledge base
- Support multiple difficulty levels
- Can be validated for compositionality

### Quick Contribution Guide

1. **🍴 Fork & Clone**
   ```bash
   git fork https://github.com/yourusername/classic-games.git
   git clone your-fork-url
   cd classic-games
   ```

2. **🔧 Setup Development Environment**
   ```bash
   python unified_env_quick_setup.py --dev
   ```

3. **🧪 Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **📝 Add Your Game & Submit a Pull Request**
   ```bash
   mkdir game_projects/your_game
   # Follow our game template structure
   ```

### 📋 **Game Implementation Checklist**
- [ ] **Strategy Knowledge Base**: Hierarchical solving strategies
- [ ] **Template System**: Fast puzzle generation patterns
- [ ] **Validation Engine**: Compositionality and uniqueness checks
- [ ] **Performance Metrics**: Benchmarking and analytics
- [ ] **Visual Processing**: Computer vision integration
- [ ] **Documentation**: Clear examples and API docs
- [ ] **Tests**: Comprehensive test coverage

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
```
MIT License - Feel free to use, modify, and distribute
Academic use encouraged - Please cite our work
Commercial use welcome - Attribution appreciated
```

## 📞 Support

For questions, suggestions, or collaboration opportunities, feel free to reach out. 

### 🎓 **Academic Citations**
If you use this project in academic research, please cite:
```bibtex
@software{classic_games_benchmark,
  title={Classic Games Benchmark: AI-Powered Puzzle Suite with Compositional Reasoning},
  author={Mahnoor Shahid & Hannes Rothe},
  institution={University of Duisburg-Essen},
  year={2025},
  url={https://github.com/yourusername/classic-games}
}
```


<div align="center">

### 🎉 **Ready to Explore AI-Powered Puzzles?**

[**🚀 Get Started**](#-quick-start) | [**📖 Documentation**](docs/) | [**🤝 Contribute**](#-contributing) | [**📊 Benchmarks**](benchmarks/)

---

**Made with ❤️ by the University of Duisburg-Essen**

</div>
