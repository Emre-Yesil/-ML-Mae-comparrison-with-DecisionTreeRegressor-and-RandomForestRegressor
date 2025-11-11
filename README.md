# 🏡 Melbourne Housing Price Prediction (Decision Tree vs Random Forest)

This project compares two regression models — **DecisionTreeRegressor** and **RandomForestRegressor** — using the **Melbourne Housing Dataset**.  
It evaluates how model complexity (controlled via `max_leaf_nodes`) and feature selection affect prediction accuracy.

---
## 📘 Overview

The script:
1. Loads the **Melbourne housing data** (`0_melb_data.csv`)
2. Defines two helper functions:
   - `getMea_DecisionTreeRegressor()`  
   - `getMea_RandomForestRegressor()`
3. Trains and validates both models using:
   - **Limited numeric features:** `Rooms`, `Bathroom`, `Landsize`, `Car`, `YearBuilt`
   - **All numeric features** from the dataset
4. Measures model performance using **Mean Absolute Error (MAE)**
5. Prints MAE values for different `max_leaf_nodes` values to help visualize **underfitting vs. overfitting** behavior.

## My output is
```cmd
DecisionTreeRegressor with limited features
for max laef nodes : 5     mae = 381474.84057758574
for max laef nodes : 50    mae = 338432.8442044053
for max laef nodes : 500   mae = 352134.5233124143
for max laef nodes : 1000  mae = 367294.7696010681
for max laef nodes : 2000  mae = 392483.108441697
for max laef nodes : 3000  mae = 405181.7156739938

DecisionTreeRegressor with full features
for max laef nodes : 5     mae = 163688.26764476238
for max laef nodes : 50    mae = 16432.98209028982
for max laef nodes : 500   mae = 1259.104490138699
for max laef nodes : 1000  mae = 824.9193783075561
for max laef nodes : 2000  mae = 721.3284241531665
for max laef nodes : 3000  mae = 721.3284241531665

Random Forest Regressor with limided features
for max laef nodes : 5     mae = 372856.701792934
for max laef nodes : 50    mae = 331297.3225493768
for max laef nodes : 500   mae = 330862.33553002577
for max laef nodes : 1000  mae = 336906.34604735376
for max laef nodes : 2000  mae = 344985.6248230969
for max laef nodes : 3000  mae = 348962.7835256929

Random Forest Regressor with full features
for max laef nodes : 5     mae = 145900.37244828756
for max laef nodes : 50    mae = 6227.1464804211755
for max laef nodes : 500   mae = 572.3124871517074
for max laef nodes : 1000  mae = 358.2807040255481
for max laef nodes : 2000  mae = 338.68942562592076
for max laef nodes : 3000  mae = 338.68942562592076
```

## DecisionTreeRegressor with limited features
   - Model struggles to learn relationships with only a few features.
   - Generalization point is around 50.
## DecisionTreeRegressor with full features
   - Way more better than limited data even the leaf size is 2000 model isn't in overfitting zone.
   - Shows that feature richness improves generalization.
## Random Forest Regressor with limided features
   - Not as good as DecisionTreeRegressor with full features but slightly better than DecisionTreeRegressor with limited features. 
   - Generalization point is around 50
## Random Forest Regressor with full features
   - Clearly the best-performing model overall.
   - Lowest MAE (~338) with max_leaf_nodes ≥ 2000
