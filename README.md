# Customer Churn Prediction (Python, SQL, XGBoost, SHAP, Streamlit)

**Author:** Rawen Jendoubi

## Problem
A telecom company loses about 26.5% of its customers. Retaining a customer is cheaper than
winning a new one, so the goal is to predict who will leave, explain why, and recommend actions.

## Approach
1. Data cleaning and EDA (pandas, seaborn) on the Telco Customer Churn dataset (7,043 customers)
2. SQL analysis (SQLite): churn by contract, tenure group, and revenue at risk (python/sql/)
3. Modeling: XGBoost, compared with a Logistic Regression baseline, 5-fold cross-validation
4. Explainability: SHAP
5. Dashboard: Streamlit app for what-if analysis per customer

## Results
| Metric | Value |
|---|---|
| AUC (test) | 0.84 |
| Logistic Regression AUC | X.XX |
| 5-fold CV AUC | X.XX |
| Recall / precision at threshold 0.5 | 52% / 66% |
| Recall / precision at threshold X | XX% / XX% |

Key drivers (SHAP): short tenure, month-to-month contract, fiber optic, high monthly charges,
electronic check payment.

![SHAP](python/reports/figures/06_shap_summary.png)
![Model evaluation](python/reports/figures/05_model_evaluation.png)

## Business recommendations
- Move new month-to-month customers to 1-2 year contracts within their first 12 months.
- Offer tech support and automatic payment to high-risk customers.
- Lower the decision threshold to catch about 70% of churners; the cost is more false alarms.

## Run it
    pip install -r requirements.txt
    cd python
    streamlit run app/streamlit_app.py

## Project structure
- python/notebooks/: analysis and model
- python/sql/: SQL queries
- python/app/: Streamlit dashboard
- python/reports/figures/: charts
- src/main/java/: optional Weka version in Java

## Limitations
Single dataset, no time dimension.

## Author
**Rawen Jendoubi**

