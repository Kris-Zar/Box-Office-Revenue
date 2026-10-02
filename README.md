# 🎬 Box Office Revenue Prediction

A machine learning project that predicts the **domestic box office revenue** of movies using an **XGBoost Regressor**. The pipeline covers thorough data cleaning, genre vectorization, log transformation for skewed features, and evaluation using Mean Absolute Error.

---

## 📌 Repository Description

> Predicts domestic box office revenue using XGBoost — with genre bag-of-words encoding, log transformation, MPAA/distributor label encoding, and StandardScaler normalization, evaluated via Mean Absolute Error on a 90/10 train-validation split.

---

## 📁 Project Structure

```
box-office-revenue-prediction/
│
├── Box_Office_Revenue.ipynb   # Main Jupyter Notebook
├── boxoffice.csv              # Input dataset (movie records)
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 📊 Dataset

The dataset (`boxoffice.csv`) contains movie-level records with financial and categorical attributes.

| Feature | Description |
|---|---|
| `title` | Movie title |
| `distributor` | Studio/distribution company |
| `domestic_revenue` | **Target variable** — domestic box office earnings |
| `opening_revenue` | Opening weekend revenue *(dropped)* |
| `world_revenue` | Worldwide revenue *(dropped)* |
| `opening_theaters` | Number of theaters at opening |
| `release_days` | Number of days in release |
| `MPAA` | Film rating (G, PG, PG-13, R, etc.) |
| `genres` | Pipe-separated genre tags |
| `budget` | Production budget *(dropped due to missing values)* |

> **Note:** `world_revenue`, `opening_revenue`, and `budget` are removed during preprocessing to avoid leakage and high missingness.

---

## ⚙️ Workflow

### 1. Data Loading
- Dataset loaded with `latin-1` encoding using `pandas`

### 2. Column Removal
- `world_revenue` and `opening_revenue` dropped (leakage risk)
- `budget` dropped due to excessive missing values

### 3. Missing Value Handling
- `MPAA` and `genres`: filled with column **mode**
- Remaining rows with nulls: dropped via `dropna()`

### 4. Data Cleaning
- `domestic_revenue`: currency symbol (`$`) stripped from string values
- `domestic_revenue`, `opening_theaters`, `release_days`: commas removed and cast to numeric

### 5. Exploratory Data Analysis (EDA)
- **MPAA distribution**: count plot
- **Mean domestic revenue by MPAA rating**: group aggregation
- **Distribution plots**: before and after log transformation for key numeric features
- **Box plots**: outlier inspection for `domestic_revenue`, `opening_theaters`, `release_days`
- **Correlation heatmap**: highlights features with correlation > 0.8

### 6. Log Transformation
- `domestic_revenue`, `opening_theaters`, `release_days` transformed using `log10` to reduce right skew

### 7. Genre Encoding
- `genres` column vectorized using `CountVectorizer` (bag-of-words)
- Each genre becomes a binary column
- Sparse genres (present in < 5% of rows) automatically dropped

### 8. Categorical Encoding
- `distributor` and `MPAA` encoded using `LabelEncoder`

### 9. Train-Validation Split
- 90% training / 10% validation
- `random_state=22`
- `title` and `domestic_revenue` excluded from features

### 10. Feature Scaling
- `StandardScaler` applied: fit on training set, transform applied to both sets

### 11. Model Training
- **Algorithm:** `XGBRegressor`
- **Max Depth:** 3
- **Learning Rate:** 0.1
- **Estimators:** 500
- **Random State:** 42

### 12. Evaluation
| Split | Metric |
|---|---|
| Training | Mean Absolute Error (MAE) |
| Validation | Mean Absolute Error (MAE) |

> Since `domestic_revenue` is log-transformed, MAE values are in log₁₀ scale.

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and log transformation |
| `matplotlib` | Subplots and figure layout |
| `seaborn` | Distribution, box, count, and heatmap plots |
| `scikit-learn` | Preprocessing, splitting, encoding, scaling |
| `xgboost` | Gradient boosted regression model |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Run the Notebook

```bash
jupyter notebook Box_Office_Revenue.ipynb
```

> **Note:** Update the hardcoded dataset path in the notebook to a relative path:
> ```python
> # Change this:
> df = pd.read_csv('C:\\Study content\\ML\\DATASETS\\boxoffice.csv', encoding='latin-1')
>
> # To this:
> df = pd.read_csv('boxoffice.csv', encoding='latin-1')
> ```

---

## 📈 Results

The model reports:
- **Training MAE** — error on the training set (log₁₀ scale)
- **Validation MAE** — error on the held-out validation set (log₁₀ scale)

Lower MAE indicates better revenue prediction accuracy.

---

## 🔮 Possible Improvements

- Fix the typo in `XGBRegressor`: `max_dapth` → `max_depth`
- Tune hyperparameters using `GridSearchCV` or `Optuna`
- Add **R² Score** and **RMSE** for additional evaluation perspective
- Reverse log transformation on predictions for interpretable MAE in dollars
- Experiment with additional models: LightGBM, CatBoost, Ridge Regression
- Explore `opening_revenue` as a feature (careful of leakage) or add it back with proper context
- Try `TfidfVectorizer` instead of `CountVectorizer` for genre encoding

---

## 📄 License

This project is for learning purposes only.
