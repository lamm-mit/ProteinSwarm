# ProteinSwarm: Swarms of Large Language Model Agents for Protein Sequence Design

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
<br>
## Summary

ProteinSwarm introduces a novel approach to protein sequence design by leveraging a swarm of Large Language Model (LLM) agents, each responsible for optimizing a single residue within a target sequence. Through multi-agent collaboration, the system iteratively refines protein sequences to achieve user-defined objectives.

## Project Structure

```
swarm/                  
├── design_loop.py                # Main design iteration engine
├── config.py                     # Configuration and data structures
├── llm_interface.py              # Agent promt
├── memory_system.py              # Memory system
├── structure_utils.py            # Structure analysis utilities
├── folding_utils.py              # Protein folding
├── structure_evaluation.py       # Design objective evaluation
├── rosetta_energy_utils.py       # Energy calculation
├── constants.py                  # Configuration constants
├── requirements.txt              # Python dependencies
├── analysis/                     # Analysis tools
│   ├── energy_monitor.py         # Energy monitor
│   ├── llm_comparison.py         # LLM comparison
│   ├── sequence_logo.R           # Sequence logo plot
│   └── sequence_space.R          # Sequence space plot
├── examples/                     # Example run scripts
│   ├── run_goals.sh              
│   └── test_design_goals.py    
└── logs/                         # Design run logs
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:lamm-mit/ProteinSwarm.git
cd ProteinSwarm

# Create virtual environment
conda create -n ProteinSwarm python=3.10
conda activate ProteinSwarm

# Install core dependencies
pip install -r requirements.txt
```
Set up [OmegaFold](https://github.com/HeliXonProtein/OmegaFold/tree/main) for structure prediction.

### 2. Configuration

Create a `local_config.py` file with your API credentials:

```python
OPENAI_API_KEY = "your-openai-api-key"
```

### 3. One-Click Design

For a quick start, use the automated pipeline:

Run a single goal with custom iterations (default 64):
```bash
python test_design_goals_batch.py 32 --goal beta_strands
```

#### Available Design Goals

- `alpha_helices_hydrophilic` - Form alpha helices using hydrophilic residues
- `beta_strands` - Form beta strands with alternating hydrophobic/polar residues
- `loose_coils` - Form loose, extended coils with polar/charged residues
- `alpha_helices_alanine_leucine_glutamate` - Form alpha helices with specific amino acids

#### Running with Energy Monitoring

Use the bash script to run all goals with real-time energy monitoring:
```bash
bash run_all_goals_with_monitoring.sh 64
```

## References

```bibtex
@article{buehler2025swarms,
  title={Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation},
  author={Wang, Fiona Y. and Lee, Di Sheng and Kaplan, David L. and Buehler, Markus J.},
  journal={arXiv preprint arXiv:submit/7027477},
  year={2025},
  url={https://arxiv.org/user/},
}
```
