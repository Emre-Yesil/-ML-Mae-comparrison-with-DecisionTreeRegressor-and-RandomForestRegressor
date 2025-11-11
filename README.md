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

