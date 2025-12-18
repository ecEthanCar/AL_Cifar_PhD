# Active Learning CIFAR-10 Simulation

Monte Carlo simulation comparing active learning strategies under noisy annotation conditions.

## Paper
**"Comparison of Active Learning Strategies for Efficient Data Annotation"**  
Ethan Carlson, Cornell University, December 2025

## Repository Contents
- `6580_al_cifar_sim.ipynb` - Main Jupyter notebook containing all experiments and analysis
- `comparison_active_learning.pdf` - Final research paper
- `AL_Results/` - Generated figures, CSVs, and pickled results
- `requirements.txt` - Python dependencies

## Key Findings
- Random sampling significantly outperforms uncertainty sampling under noise (Cohen's d = 2.67, p = 0.001)
- Uncertainty sampling actively "poisons" training with mislabeled boundary examples
- 68% of performance variance explained by strategy choice (η² = 0.680)

## Notebook Structure

### Cell 1: Setup & Reproducibility
- Google Drive mounting for Colab
- Seed management functions
- CRN infrastructure

### Cell 2: Core Implementation
- `SimpleCNN` model architecture
- Training/evaluation functions
- Noisy oracle (`noisy_label`)
- Query strategies: `query_random`, `query_uncertainty`, `query_diversity_kmeans`
- `NoisyLabelDataset` wrapper
- `run_single_simulation` - main AL loop
- `run_experiment` - orchestrates multiple replications

### Cell 3: Quick Validation
- Single replication test run
- ε = 0%, N = 1
- Verifies end-to-end pipeline

### Cell 4: Main Experiment (Fixed Noise)
- ε ∈ {0%, 5%, 10%, 15%}
- 5 replications per condition
- 3 strategies × 4 noise levels × 5 reps = 60 runs
- **Runtime**: ~7.7 hours on Colab T4 GPU

### Cell 5: Visualization & Export
- Generates learning curves (PNG)
- Exports summary statistics (CSV, JSON)
- Computes mean ± 95% CI across replications

### Cell 6: AULC Extraction
- Loads pickled results
- Constructs long-format DataFrame (1,200 rows)
- Exports `aulc_results.csv`

### Cell 7: Statistical Analysis
- One-way ANOVA per noise level
- Tukey HSD post-hoc tests
- Effect sizes (η², Cohen's d)
- **Key Output**: Table 3 in paper

### Cells 8-11: Exploratory Experiments
- Secondary/tertiary robustness checks
- Smaller-scale validation runs
- Not included in final paper

### Cell 12: Robust Stochastic Run
- ε ~ Uniform[0%, 15%] drawn per replication
- **This is the headline result**
- 5 replications, full CIFAR-10
- **Runtime**: ~2.5 hours

### Cell 13-14: Stochastic Analysis
- Visualization of stochastic results (Figure 2)
- ANOVA on stochastic condition
- **Generates**: η² = 0.680, d = 2.67

## Experiments Overview

| Experiment | Epsilon | Reps | Purpose | Runtime |
|-----------|---------|------|---------|---------|
| **Fixed Noise** | {0, 0.05, 0.10, 0.15} | 5 | Dose-response | 7.7 hrs |
| **Stochastic** | U[0, 0.15] | 5 | Robustness | 2.5 hrs |
| Tertiary | U[0, 0.15] | 3 | Validation | 0.5 hrs |

## Implementation Details
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test)
- **Model**: SimpleCNN (3 conv blocks → 128-d embedding → 10 classes)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Training**: 10 epochs per acquisition round, model reinitialized each round
- **Augmentation**: RandomCrop(32, padding=4) + RandomHorizontalFlip
- **Variance Reduction**: Common Random Numbers (CRN)
  - Synchronized: L₀ selection, θ₀ initialization, mini-batch order, oracle noise
  - Per-replication seeds ensure reproducibility
- **Initial Pool**: 100 samples (stratified, 10 per class)
- **Budget**: 2,000 labels (19 acquisition rounds)
- **Query Batch**: 100 samples per round

## Requirements
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch >= 2.0.0` (GPU strongly recommended)
- `torchvision >= 0.15.0`
- `scikit-learn >= 1.2.0`
- `numpy, pandas, matplotlib, scipy, statsmodels`

## Usage

### Option 1: Google Colab (Recommended)
1. Upload `6580_al_cifar_sim.ipynb` to Colab
2. Run Cell 1 to mount Google Drive
3. Execute cells sequentially
4. Results save to `/content/drive/MyDrive/AL_Results/`

### Quick Start (Test Run)
Execute only Cells 1-3 for a rapid validation (~15 minutes).

### Reproduce Paper Results
Execute Cells 1-2, 4-5, 7, 12-14 (~10 hours total).

## Results
See `AL_Results/` directory structure:
```
AL_Results/
├── al_results_e{0,5,10,15}.pkl          # Raw simulation data (fixed)
├── summary_e{0,5,10,15}.{json,csv}      # Aggregated statistics
├── al_learning_curves_e{0,5,10,15}.png  # Figure 1 (4 subplots)
├── aulc_results.csv                      # AULC per replication
├── robust_stochastic/
│   ├── al_results_e15.pkl               # Stochastic raw data
│   ├── summary_e15.{json,csv}           # Stochastic statistics
│   └── al_learning_curves_e15.png       # Figure 2
└── [tertiary/, secondary/]              # Exploratory runs
```

## Reproducing Key Tables & Figures

| Paper Element | Notebook Cells | Output File |
|--------------|---------------|-------------|
| **Figure 1** (Fixed noise curves) | 4 → 5 | `al_learning_curves_e{0,5,10,15}.png` |
| **Figure 2** (Stochastic curve) | 12 → 13 | `robust_stochastic/al_learning_curves_e15.png` |
| **Table 3** (ANOVA) | 7, 14 | Console output |
| **Table 2** (AULC means) | 14 | `aulc_results.csv` + console |

## Key Code Locations

**Core Functions:**
- Stratified sampling: Cell 2, lines ~30-35
- Noisy oracle: Cell 2, lines ~92-97
- Uncertainty query: Cell 2, lines ~105-120
- Diversity query: Cell 2, lines ~140-165
- CRN setup: Cell 2, lines ~196-197, 205-206
- Main AL loop: Cell 2, `run_single_simulation`

**Critical Hyperparameters:**
- Budget/batch: Cell 2, config dictionaries (e.g., Cell 4, lines ~10-15)
- Model architecture: Cell 2, `SimpleCNN.__init__`
- Optimizer: Cell 2, line ~309 in `run_single_simulation`

## Troubleshooting (If Issues Encountered)

**Out of Memory:**
- Reduce `subset` in config (e.g., `'subset': 10000`)
- Use smaller models in exploratory runs (Cell 8+)

**Slow Runtime:**
- Ensure GPU is enabled in Colab: Runtime → Change runtime type → T4 GPU
- Reduce `replications` to 3 for testing
- Use `max_epochs=5` instead of 10

**Reproducibility Issues:**
- Verify `seed` is set in config
- Check `torch.backends.cudnn.deterministic = True` in Cell 1
- Ensure same PyTorch/CUDA versions


## License
MIT License - Academic use encouraged. Please cite if used in publications.

## Contact
Ethan Carlson - Cornell University  
Questions/issues: Open a GitHub issue or contact via Cornell email.

---

**Note**: Runtime estimates based on Google Colab T4 GPU. CPU execution may take significantly longer.
