# Intro to Machine Learning — İstanbul Bilgi (2-day, Python/Colab)

Reproduce-then-extend tutorials for the 2-day course. Each notebook reproduces a published study's
regression and then extends it with machine learning, evaluated honestly out of sample.

See the **[Codebook & Application Guide](CODEBOOK.md)** for each study, its data, variables, and methods.

**Runs in Google Colab with one click** — the notebooks read their data from this repo's raw links, so
nothing needs to be uploaded. Clicking a badge opens the notebook in **your own** Colab session (whatever
Google account you are signed into); it never runs in anyone else's account. Notebooks that use `shap` or
`abess` install them in a first setup cell.

## Open a tutorial in Colab

| Day | Tutorial | Launch |
|-----|----------|--------|
| Day 1 | Voter turnout & term limits — OLS + out-of-sample prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/01_turnout_day1.ipynb) |
| Day 2 | Civil-war onset (Fearon–Laitin) — SVM, k-NN & random forest + SHAP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/02_fearon_laitin_day2.ipynb) |
| Day 2 | Militarized disputes — RF/XGBoost + SHAP (RSF) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/05_gibler_braithwaite_day2.ipynb) |
| Day 2 | Pinyon Jay habitat — RF/XGBoost + SHAP + partial dependence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/06_pinyonjay_rsf_day2.ipynb) |
| Day 2 | Snake occurrence — RF/XGBoost + SHAP + partial dependence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/07_snake_rsf_day2.ipynb) |

| Day 1 | Economic shocks & regional elite splits — the Catalan Lliga (Vall-Prat 2022) — regularization + ABESS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/11_economic_shocks_lliga_day1.ipynb) |
| Day 1 | Election-law reform after 2000 (Palazzolo & Moscardelli) — regularization + ABESS (49-state design) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/desmarais-lab/desmarais-lab.github.io/blob/master/istanbul_bilgi_ml_files/notebooks/12_election_law_reform_day1.ipynb) |

## Contents
- `notebooks/` — the ten tutorials (Python).
- `data/` — datasets, read automatically via raw GitHub links.
- `slides/` — the Day 1 and Day 2 slide decks (PDF).

## Running locally
```bash
pip install numpy pandas matplotlib statsmodels scikit-learn xgboost shap abess openpyxl
```
Then open any notebook in Jupyter.
