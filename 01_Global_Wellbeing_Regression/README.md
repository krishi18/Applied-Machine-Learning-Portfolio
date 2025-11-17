# Global Well-being & Health Regression Analysis 🌍🏥

## 📖 Project Overview
This project applies supervised machine learning techniques to solve two distinct socio-economic regression problems: predicting **World Happiness Scores** and estimating **Life Expectancy**.

Using a modular pipeline approach, I explored how economic factors (like GDP) and health metrics (like mortality rates) non-linearly correlate with human quality of life. The goal was to minimize prediction error (RMSE) by testing Linear, Polynomial, and Regularized regression models.

## 📂 Directory Contents
* **`01_Happiness_Prediction.ipynb`**: Regression analysis on the World Happiness Report dataset.
* **`02_Life_Expectancy_Analysis.ipynb`**: Regression analysis on WHO Life Expectancy data.

---

## 📊 Part 1: World Happiness Prediction
**Objective:** Predict the "Life Ladder" score (Happiness Index) for countries based on macro-economic factors.

### Key Techniques
* **Feature Engineering:** Applied `PolynomialFeatures (degree=2)` to capture the diminishing returns of wealth on happiness (i.e., more money increases happiness, but only up to a point).
* **Handling Skew:** Addressed right-skewed data in `GDP_per_capita` and `Social_support`.
* **Model Competition:**
    * *Linear Regression (Baseline)*
    * *Polynomial Regression* (🏆 **Winner**: Captured non-linear complexities best)
    * *Stochastic Gradient Descent (SGD)* (Used for robust iterative learning)

---

## 🏥 Part 2: Life Expectancy Prediction
**Objective:** Estimate average life expectancy based on immunization coverage, mortality rates, and economic resources.

### Key Insights
* **Advanced Imputation:** Implemented **Grouped Median Imputation** for handling missing values. instead of filling missing data (e.g., `Hepatitis B`, `GDP`) with a global average, I filled them using the median value **specific to each Country**. This preserved the distinct socio-economic profile of each nation.
* **Feature Importance:** Correlation analysis revealed that `Schooling` and `Income Composition` were the strongest positive predictors, while `Adult Mortality` had the strongest negative correlation.
* **Regularization:** Utilized `Ridge` regression to penalize large coefficients and prevent overfitting when using many health indicators.

---

## 🛠️ Technical Approach (Common Pipeline)
Both projects utilize a strict Scikit-Learn pipeline to prevent data leakage:
1.  **Preprocessing:**
    * **Custom Imputation:** utilized Grouped Median strategies to respect categorical boundaries (Country/Region) before falling back to global medians.
    * **Scaling:** Applied `StandardScaler` to normalize features, ensuring distance-based algorithms (like SGD/Ridge) converged correctly.
2.  **Validation:**
    * Used **K-Fold Cross-Validation** to verify model stability across different data subsets.
    * Evaluated using **Root Mean Squared Error (RMSE)** to penalize larger prediction errors.

## 🚀 How to Run
Ensure you have the necessary libraries installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
