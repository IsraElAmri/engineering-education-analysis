# Engineering Education Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Hypothesis%20Testing-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python analysis pipeline developed as an individual module within a 12-person student team project at the National Engineering School of Tunis (ENSIT). The wider team project evaluated the impact of artificial intelligence integration across engineering education models.

This repository isolates my specific contribution: the modular Python analysis pipeline implementing non-parametric statistical hypothesis testing, unsupervised clustering (K-Means), and supervised feature importance ranking (Random Forest) evaluated on simulated curriculum data.

## Demonstration Data Notice

To facilitate self-contained execution without distributing proprietary survey responses, this repository ships with a 6-row demonstration dataset (`data/sample_before_ai.csv` and `data/sample_after_ai.csv`).

- The sample files serve exclusively to verify pipeline execution and format compatibility.
- The visualizations in `figures/` originate from the earlier run on the full simulated dataset.
- Every metric reported in project documentation was evaluated on the simulated dataset; no real-world empirical conclusions are drawn from the 6-row demonstration sample.

## Pipeline Architecture

The pipeline (`src/education_analyzer.py`) processes baseline and post-AI curricular indicators:

1. **Preprocessing**: Validates column schemas, standardizes currency values, and normalizes numeric indicators using `StandardScaler`.
2. **Statistical Hypothesis Testing**: Conducts paired comparison tests comparing pre- and post-AI metrics to evaluate shifts in AI course percentage and transition indicators.
3. **Unsupervised Clustering**: Applies K-Means clustering ($k=3$) to group educational profiles based on structural indicators (starting salary, annual cost, AI course share).
4. **Supervised Feature Importance**: Fits a Random Forest classifier to identify which curricular and economic features best discriminate between educational models on simulated data.
5. **Reporting**: Generates summary CSV tables and an automated HTML diagnostic report.

## Baseline Observations (Simulated Dataset)

Across the simulated project run, the analysis pipeline identified:
- **Top Differentiators**: Annual tuition/local cost and graduate starting salary accounted for the dominant share of relative feature importance in separating institutional models on the simulated dataset.
- **Curriculum Shifts**: AI course share increased between pre- and post-integration periods across all modeled institutions in the simulation.

## Repository Structure

```
engineering-education-analysis/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── sample_after_ai.csv
│   └── sample_before_ai.csv
├── figures/
│   ├── correlation_heatmap.png
│   └── feature_importance.png
└── src/
    ├── education_analyzer.py
    ├── run_analysis.py
    └── test_education_analyzer.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/IsraElAmri/engineering-education-analysis.git
cd engineering-education-analysis
pip install -r requirements.txt
```

### Running the Analysis

Execute the end-to-end pipeline on the demonstration sample:

```bash
python src/run_analysis.py
```

Generated outputs will be saved to `figures/` and `phase1_analysis_report.html`.

### Running Unit Tests

Run the test suite covering data loading, preprocessing, clustering, and report generation:

```bash
python -m unittest discover -s src -p "test_*.py"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
