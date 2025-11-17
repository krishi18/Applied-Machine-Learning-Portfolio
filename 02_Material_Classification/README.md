# Geological & Material Classification System 🪨🔍

## 📖 Project Overview
This project addresses the challenge of automated material identification in physical sciences. Using the **UCI Glass Dataset** and an **Aggregate Rock Dataset**, I built multi-class classification models to distinguish between complex physical materials based on their chemical and textural properties.

A key highlight of this project is the **Human-Level Performance Benchmark**, where machine learning models were evaluated not just on theoretical accuracy, but compared directly against human classification trials to test viability for real-world automation.

## 📂 Directory Contents
* **`01_Rock_Classification.ipynb`**: Classification of geological samples compared against human trials.
* **`02_Glass_Identification.ipynb`**: Forensic identification of glass types using chemical analysis.

---

## 🪨 Part 1: Rock Type Classification
**Objective:** Classify aggregate rock samples into geological categories (Igneous, Metamorphic, Sedimentary) using visual feature data.

### Key Innovation: Human Benchmarking
* **The Challenge:** Rock classification is subjective. I compared my model's accuracy against a dataset of **human trials** (`trialData.csv`) to see if ML could outperform human observers.
* **Result:** The ensemble models achieved comparable or superior consistency compared to average human classification rates, validating the potential for automated geological sorting.

### Modeling Approach
* **Voting Classifier:** Combined `LogisticRegression`, `SVC`, and `RandomForest` into a generic Soft Voting ensemble to stabilize predictions.
* **Model Selection:**
    * *Random Forest:* Best handled the discrete "feature presence" data.
    * *Support Vector Machine (SVM):* Tuned using `GridSearchCV` to find optimal hyperplanes for separating rock textures.

---

## 🧪 Part 2: Forensic Glass Identification
**Objective:** Identify the source of glass splinters (e.g., Building Windows, Vehicle Headlamps, Containers) based on Refractive Index (RI) and chemical elements (Na, Mg, Al, Si).

### Key Insights
* **Non-Linearity:** Linear models (Logistic Regression) failed significantly, proving that the relationship between chemical elements and glass type is highly non-linear.
* **Class Imbalance:** The dataset was heavily skewed (many window samples, few headlamps).
    * *Solution:* I utilized **Stratified K-Fold** validation and `class_weight='balanced'` parameters in SVM to ensure rare glass types were not ignored.
* **Winner:** The **RBF-Kernel SVM** and **Random Forest** were the only models capable of capturing the complex chemical boundaries needed for forensic accuracy.

---

## 🛠️ Technical Approach (Common Pipeline)
1.  **Data Preprocessing:**
    * **Standardization:** Applied `StandardScaler` rigorously. This was critical for the SVM models (Glass & Rock) which are distance-sensitive.
    * **Imbalance Handling:** Used Stratified splitting to maintain class ratios during training/testing.
2.  **Hyperparameter Tuning:**
    * Utilized `GridSearchCV` to optimize $C$ and $\gamma$ (Gamma) parameters for SVM, and `n_estimators`/`max_depth` for Random Forests.
3.  **Evaluation Metrics:**
    * **Confusion Matrices:** Used to visualize exactly *which* classes were being confused (e.g., confusing "Float Glass" with "Non-Float Glass").
    * **Pearson Correlation:** Analyzed feature redundancy (e.g., correlation between Refractive Index and Calcium).

## 🚀 How to Run
Ensure you have the necessary libraries installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib scipy
