# IPL First-Innings Score Prediction 

A machine learning project that predicts the first-innings score of an IPL match using historical ball-by-ball data (2008–2017).

> **Note:** This project predicts the **first-innings total score**, not the match winner. Score prediction is treated as a regression problem based on match state (overs, wickets, run rate).

##  Overview

Using ~76,000 ball-by-ball records from IPL seasons 1–10, this project trains and compares three regression models to estimate the final first-innings score at any point in an ongoing match.

##  Dataset

- **Source:** [IPL Dataset (Season 2008–2017) — Kaggle](https://www.kaggle.com/yuvrajdagur/ipl-dataset-season-2008-to-2017)
- **Records:** 76,014 ball-by-ball entries
- **Key columns:** `batting_team`, `bowling_team`, `runs`, `wickets`, `overs`, `runs_last_5`, `wickets_last_5`, `total`

##  Data Preprocessing

- Removed irrelevant columns (`mid`, `date`, `venue`, `batsman`, `bowler`, `striker`, `non-striker`)
- Filtered to 8 consistently active franchises to avoid noise from rebranded/discontinued teams
- Removed the first 5 overs of each innings (too volatile to be predictive)
- One-hot encoded `batting_team` and `bowling_team`

##  Models Trained

| Model | Type |
|---|---|
| Decision Tree Regressor | Baseline |
| Random Forest Regressor | Ensemble |
| Neural Network | Deep Learning |

Each model was trained on match-state features (current runs, wickets, overs, last-5-overs stats, team identity) to predict the final first-innings total.

##  Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run the notebook
1. Download the dataset from the Kaggle link above and save it as `matches.csv`
2. Open `IPL_Prediction_Model_Training.ipynb` in Jupyter or Google Colab
3. Upload `matches.csv` when prompted (or place it in the working directory)
4. Run all cells

### Use a trained model
```python
from joblib import load

model = load('forest_model.pkl')  # or tree_model.pkl / neural_nets_model.pkl
score = predict_score('Mumbai Indians', 'Kings XI Punjab',
                       overs=12.3, runs=113, wickets=2,
                       runs_last_5=55, wickets_last_5=0)
print(f"Predicted Score: {score}")
```

##  Results

The models were validated against real historical match outcomes. Example:

| Batting Team | Bowling Team | Predicted Score | Actual Score |
|---|---|---|---|
| Mumbai Indians | Kings XI Punjab | 189 | 176 |
| Kolkata Knight Riders | Chennai Super Kings | 175 | 172 |
| Delhi Daredevils | Mumbai Indians | 107 | 110 |

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib (model persistence)

##  Future Improvements

- Incorporate player-level features (batting/bowling form) using squad data
- Extend to a second-innings chase-prediction model
- Build a classifier for match-winner prediction using match-level (not ball-by-ball) data

##  License

This project is for educational purposes as part of a data science internship.
