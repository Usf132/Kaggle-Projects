# 🚢 Titanic – Machine Learning from Disaster

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-Unspecified-lightgrey)

A machine learning notebook that predicts passenger survival on the RMS Titanic using the classic [Kaggle "Titanic: Machine Learning from Disaster"](https://www.kaggle.com/competitions/titanic) dataset. The project builds a custom `scikit-learn` preprocessing pipeline and a tuned **Random Forest Classifier** to generate survival predictions for the Kaggle test set.

---

## ✨ Features

- 📊 **Exploratory data analysis** — correlation heatmap of numeric features (`Survived`, `Pclass`, `Age`, `Fare`, etc.).
- 🎯 **Stratified train/test split** using `StratifiedShuffleSplit` (stratified on `Survived`, `Pclass`, and `Sex`) to preserve class balance.
- 🧩 **Custom `scikit-learn` transformers**, chained in a `Pipeline`:
  - `AgeImputer` — fills missing `Age` values with the column mean.
  - `FeatureEncoder` — one-hot encodes `Embarked` and `Sex`.
  - `FeatureDropper` — drops non-numeric / redundant columns (`Name`, `Ticket`, `Cabin`, `Embarked`, `Sex`, `Embarked_nan`).
- ⚖️ **Feature scaling** with `StandardScaler`.
- 🌲 **Model training & hyperparameter tuning** — `RandomForestClassifier` optimized via `GridSearchCV` (3-fold CV) over `n_estimators`, `max_depth`, and `min_samples_split`.
- ✅ **Held-out evaluation** — accuracy reported on a stratified test split before retraining on the full dataset.
- 🏁 **Final production model** — retrained on 100% of `train.csv` and used to generate predictions for `test.csv`.
- 📄 **Kaggle-ready submission file** — `data/predictions.csv` containing `PassengerId` and predicted `Survived`.

> **Note:** The `images/` folder contains pre-generated charts (confusion matrix, feature importance, age distributions, correlation heatmap, split distribution), but the current notebook code only produces the correlation heatmap and the train/test histogram plot. The code that generated the confusion matrix, feature importance, and age-distribution images is **not present** in the notebook — these appear to be outputs from a prior version of the analysis and are included here as-is rather than assumed to be reproducible.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Environment | Jupyter Notebook |
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (Pipeline, RandomForestClassifier, GridSearchCV, StandardScaler, OneHotEncoder, SimpleImputer, StratifiedShuffleSplit) |

No web framework, API server, database, or front end is present in this project — it is a self-contained data science notebook.

---

## 📁 Project Structure

```
Titanic - Machine Learning from Disaster/
├── Titanic - Machine Learning from Disaster.ipynb   # Main notebook: EDA, pipeline, training, prediction
├── data/
│   ├── train.csv                # Labeled training data (Kaggle)
│   ├── test.csv                 # Unlabeled test data (Kaggle)
│   ├── gender_submission.csv    # Kaggle's sample submission file
│   └── predictions.csv          # Generated predictions (notebook output)
└── images/
    ├── correlation_heatmap.png       # Reproducible from notebook code
    ├── split_distribution.png        # Reproducible from notebook code
    ├── confusion_matrix.png          # Present, but no generating code in notebook
    ├── feature_importance.png        # Present, but no generating code in notebook
    ├── age_distribution_all.png      # Present, but no generating code in notebook
    ├── age_distribution_analysis.png # Present, but no generating code in notebook
    ├── age_distribution_pclass_1.png # Present, but no generating code in notebook
    ├── age_distribution_pclass_2.png # Present, but no generating code in notebook
    └── age_distribution_pclass_3.png # Present, but no generating code in notebook
```

---

## 📦 Requirements

There is no `requirements.txt` or `environment.yml` in the project. Based on the imports used in the notebook, the following packages are required:

- `python` (3.x)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter` (to run the `.ipynb` file)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd "Titanic - Machine Learning from Disaster"
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   No dependency file is included in the project, so install the packages manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook "Titanic - Machine Learning from Disaster.ipynb"
   ```

---

## 🚀 Usage

Run the notebook cells sequentially from top to bottom:

1. **Load & explore data** — `data/train.csv` is loaded into a `pandas` DataFrame; a correlation heatmap is plotted.
2. **Stratified split** — the training data is split into a stratified train/test set (80/20) based on `Survived`, `Pclass`, and `Sex`.
3. **Build the preprocessing pipeline** — `AgeImputer → FeatureEncoder → FeatureDropper` transforms raw passenger data into a fully numeric feature matrix.
4. **Train & tune the model** — a `RandomForestClassifier` is tuned with `GridSearchCV` on the training split; accuracy is measured on the held-out test split.
5. **Retrain on full data** — the pipeline and model are refit on the entire `train.csv` dataset ("production" model).
6. **Predict on Kaggle's test set** — `data/test.csv` is transformed with the fitted pipeline and scaler, and survival predictions are generated.
7. **Export submission** — predictions are written to `data/predictions.csv` in `PassengerId, Survived` format, ready for Kaggle submission.

---

## 🔧 Configuration

There are no external configuration files, environment variables, or CLI arguments in this project. All parameters (e.g. the `GridSearchCV` hyperparameter grid, the 80/20 split ratio, `random_state=42`) are hard-coded directly in the notebook cells.

---

## 🔌 API Endpoints

Not applicable — this project is a data analysis/ML notebook with no API, server, or web service component.

---

## 📸 Screenshots

Reproducible directly from the notebook code:

| Correlation Heatmap | Train/Test Split Distribution |
|---|---|
| `images/correlation_heatmap.png` | `images/split_distribution.png` |

Present in `images/` but not generated by any code currently in the notebook (included for completeness, not verified as reproducible):

| Confusion Matrix | Feature Importance |
|---|---|
| `images/confusion_matrix.png` | `images/feature_importance.png` |

| Age Distribution (All) | Age Distribution by Pclass |
|---|---|
| `images/age_distribution_all.png` | `images/age_distribution_pclass_1.png`, `_2.png`, `_3.png` |

*(Add the actual image previews here, e.g. `![Correlation Heatmap](images/correlation_heatmap.png)`, once the README is placed in the project root.)*

---

## 💡 Example Usage

Output of the model evaluation cell on the stratified held-out split:

```
Best estimator: RandomForestClassifier(max_depth=5, min_samples_split=4)
Test accuracy: 0.8324022346368715
```

Sample of the generated `data/predictions.csv`:

```csv
PassengerId,Survived
892,0
893,0
894,0
895,0
896,0
...
```

---

## 🔭 Future Improvements

- Add a `requirements.txt` or `environment.yml` to pin dependencies and Python version.
- Add the actual code used to generate `confusion_matrix.png`, `feature_importance.png`, and the age-distribution plots, or remove them if outdated.
- Extract the notebook logic into reusable Python modules/scripts (e.g. `preprocessing.py`, `train.py`, `predict.py`) for better maintainability outside of Jupyter.
- Replace the `Age` mean-imputation strategy with a more informed approach (e.g. median imputation grouped by `Pclass`/`Sex`, or model-based imputation).
- Add cross-validated evaluation metrics beyond accuracy (precision, recall, F1, ROC-AUC), especially given class imbalance in `Survived`.
- Add unit tests for the custom transformers (`AgeImputer`, `FeatureEncoder`, `FeatureDropper`).
- Consider feature engineering from currently dropped columns (e.g. deriving `Title` from `Name`, or `Deck` from `Cabin`) instead of discarding them outright.
- Persist the trained pipeline/model (e.g. with `joblib`) so predictions can be generated without rerunning the full notebook.
- Add a `.gitignore` for generated artifacts like `data/predictions.csv`.
- Fix the deprecated `fillna(method="ffill")` call (flagged by a `FutureWarning` in the notebook output) in favor of `.ffill()`.

---

## 🤝 Contributing

No `CONTRIBUTING.md` or contribution guidelines are currently present in the project. If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes.
4. Open a pull request describing your changes.

---

## 📄 License

No license file (e.g. `LICENSE` or `LICENSE.md`) is currently included in the project. Without an explicit license, default copyright law applies and the code is **not** open for reuse by others. Consider adding an [MIT](https://choosealicense.com/licenses/mit/) or [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) license if you intend to share this publicly.

---

## 👤 Author

Author information is not present in the project files. Add your name, GitHub profile, and/or contact details here.

---

## 📝 Suggested Improvements

**Project:**
- Introduce dependency and environment management (`requirements.txt`, `environment.yml`, or `pyproject.toml`).
- Modularize the notebook into scripts for a cleaner, more testable, production-style ML workflow.
- Add a `LICENSE` file and author/contact details.
- Reconcile the `images/` folder with the notebook — either regenerate the missing plots' source code or remove unused images to avoid confusion.
- Fix the deprecation warning in the `fillna(method="ffill")` call.
- Track experiments/metrics (e.g. with MLflow or a simple results log) so different model runs and hyperparameters are comparable over time.

**README:**
- Once this file is placed in the project root, embed the actual images (e.g. `![Correlation Heatmap](images/correlation_heatmap.png)`) instead of placeholder table references.
- Add a live badge for a CI workflow if one is introduced (e.g. notebook execution checks via `nbconvert --execute`).
- Link directly to the Kaggle competition page for dataset context and leaderboard comparison.
- If the project is made public with a chosen license, update the License section and badge accordingly.
