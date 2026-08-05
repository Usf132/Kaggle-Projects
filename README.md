# 📊 Kaggle-Projects

A collection of end-to-end machine learning notebooks built for classic [Kaggle](https://www.kaggle.com/) competitions. Each project lives in its own folder with its own notebook, data, generated artifacts, and README — covering the full pipeline from raw data to a Kaggle-ready submission file.

## 📂 Projects

| Project | Task | Approach | Notebook |
|---|---|---|---|
| 🏠 [House Prices - Advanced Regression Techniques](./House%20Prices%20-%20Advanced%20Regression%20Techniques) | Predict residential home sale prices (regression) | Tuned **XGBoost Regressor** + **Keras ANN**, with missing-value imputation, one-hot encoding, and outlier removal | `code.ipynb` |
| 🚢 [Titanic - Machine Learning from Disaster](./Titanic%20-%20Machine%20Learning%20from%20Disaster) | Predict passenger survival (classification) | Custom **scikit-learn Pipeline** (imputation, encoding, feature dropping) + tuned **Random Forest Classifier** | `Titanic - Machine Learning from Disaster.ipynb` |

Each folder is self-contained: open its own README for a full breakdown of features, project structure, requirements, and usage instructions.

## 🛠️ Tech Stack

Across the repo, the notebooks make use of:

- **Language:** Python 3
- **Environment:** Jupyter Notebook
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Classical ML:** scikit-learn (Pipelines, GridSearchCV / RandomizedSearchCV, StandardScaler)
- **Gradient Boosting:** XGBoost
- **Deep Learning:** TensorFlow / Keras

## 🚀 Getting Started

Each project is independent and has its own data and dependencies. To run one:

```bash
# Clone the repository
git clone https://github.com/Usf132/Kaggle-Projects.git
cd Kaggle-Projects

# Move into the project you want to run
cd "House Prices - Advanced Regression Techniques"   # or "Titanic - Machine Learning from Disaster"

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies (see each project's README for the full list)
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch Jupyter and open the notebook
jupyter notebook
```

> ⚠️ Neither project currently includes a `requirements.txt` or environment file — see each project's README for the exact packages to install.

## 📁 Repository Structure

```
Kaggle-Projects/
├── House Prices - Advanced Regression Techniques/
│   ├── code.ipynb
│   ├── data/
│   ├── images/
│   ├── model/
│   └── outputs/
├── Titanic - Machine Learning from Disaster/
│   ├── Titanic - Machine Learning from Disaster.ipynb
│   ├── data/
│   └── images/
├── LICENSE
└── README.md
```


This repository is licensed under the [MIT License](./LICENSE).
