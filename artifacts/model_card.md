# Model Card — Walmart Weekly Sales Predictor

## Model Purpose
This model predicts the weekly sales revenue for a given Walmart store based on historical sales
patterns, calendar features, and external economic indicators. It is intended for use by retail
analysts and store operations teams to support inventory planning, staffing decisions, and
promotional scheduling.

## Training Data
- **Source:** Walmart_Sales.csv — publicly available Walmart store sales dataset
- **Date range:** 2010-02-05 to 2012-10-26
- **Records:** ~6,390 rows after feature engineering (45 stores × ~142 weeks, minus first week per store)
- **Stores:** 45 unique store locations
- **Features used:**
  - `Store` — store identifier
  - `Year`, `Month`, `Week` — calendar time features extracted from Date
  - `Is_Quarter_End` — binary flag (1 if month is 3, 6, 9, or 12)
  - `Holiday_Flag` — binary flag for US holiday weeks
  - `Temperature` — average regional temperature (°F)
  - `Fuel_Price` — regional fuel price (USD/gallon)
  - `CPI` — Consumer Price Index
  - `Unemployment` — regional unemployment rate
  - `Sales_Lag1` — previous week's sales for the same store
  - `Sales_Rolling4` — 4-week trailing average of past sales (weeks t-1 through t-4)
- **Train/test split:** Chronological 80/20 — data sorted by Date across all stores, then first
  80% of rows (earlier weeks) used for training and last 20% (later weeks) for testing.
  No random shuffling; temporal ordering is preserved so the model never sees future data during training.

## Performance Metrics

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 68042.47 | 49840.53 | 0.98 |
| Random Forest | 75721.36 | 53702.12 | 0.98 |

**Winner: Linear Regression** — selected based on lower RMSE with equivalent R².

## Feature Importance (Random Forest reference)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Sales_Rolling4 | 0.6364 |
| 2 | Sales_Lag1 | 0.3057 |
| 3 | Week | 0.0403 |
| 4 | Temperature | 0.0028 |
| 5 | CPI | 0.0027 |

## Limitations
1. **Limited time range:** Trained on 2010–2012 data only. Sales patterns, economic conditions, and
   consumer behavior have changed significantly since then.
2. **Lag features cause first-week data loss:** The first recorded week for each store cannot be
   predicted because `Sales_Lag1` and `Sales_Rolling4` require prior weeks' data.
3. **No store metadata:** Store size, location, demographics, and local competition are not included.
   Two stores with different sizes may have identical feature values but very different actual sales.
4. **Economic features are regional averages:** Temperature, CPI, and Unemployment are regional
   averages and may not accurately reflect hyper-local conditions.
5. **No external events:** The model has no awareness of major events (extreme weather, supply chain
   disruptions, competitor openings) that can dramatically affect sales.

## Ethical Considerations
1. **Not suitable for staffing decisions alone:** Predictions should not directly drive hiring or
   scheduling without human review, as systematic underestimation for certain stores could lead
   to understaffing.
2. **Demographic blind spots:** The model has no visibility into the demographic composition of a
   store's catchment area. Predictions may be systematically less accurate for stores serving
   underrepresented or economically vulnerable communities.
3. **Temporal fairness:** A model trained on 2010–2012 data reflects the post-2008 recession
   recovery period. Applying it to current data without retraining may introduce systematic bias.

## How to Use

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/model.pkl")

new_data = pd.DataFrame([{
    "Store": 1,
    "Year": 2012,
    "Month": 12,
    "Week": 50,
    "Is_Quarter_End": 1,
    "Holiday_Flag": 1,
    "Temperature": 35.2,
    "Fuel_Price": 3.45,
    "CPI": 212.5,
    "Unemployment": 7.8,
    "Sales_Lag1": 1_500_000,
    "Sales_Rolling4": 1_450_000,
}])

prediction = model.predict(new_data)
print(f"Predicted Weekly Sales: ${prediction[0]:,.0f}")
```
