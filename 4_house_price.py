import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def getMea_RandomForestRegressor(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    prediction_val = model.predict(val_X)
    mea = mean_absolute_error(val_y, prediction_val)
    return mea

def getMea_DecisionTreeRegressor(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    prediction_val = model.predict(val_X)
    mea = mean_absolute_error(val_y, prediction_val)
    return mea

try:
    #data
    melbourne_data = pd.read_csv("0_melb_data.csv")
    #price
    y=melbourne_data.Price

    #with full features
    X_full_fearures = melbourne_data.select_dtypes(include=['number'])
    train_X_full, val_X_full, train_y_full, val_y_full = train_test_split(X_full_fearures, y, random_state=1)

    #with limited features
    X_lim_fearures = ['Rooms', 'Bathroom', 'Landsize','Car','YearBuilt']
    X_limited = melbourne_data[X_lim_fearures]
    train_X_lim, val_X_lim, train_y_lim, val_y_lim = train_test_split(X_limited, y, random_state=1)

    print("\nDecisionTreeRegressor with limited features")
    for max_leaf_nodes in [5, 50, 500, 1000, 2000, 3000]:
       mae = getMea_DecisionTreeRegressor(max_leaf_nodes, train_X_lim, val_X_lim, train_y_lim, val_y_lim)
       print(f"for max laef nodes : {max_leaf_nodes:<5} mae = {mae}")
       
    print("\nDecisionTreeRegressor with full features")
    for max_leaf_nodes in [5, 50, 500, 1000, 2000, 3000]:
       mae = getMea_DecisionTreeRegressor(max_leaf_nodes, train_X_full, val_X_full, train_y_full, val_y_full)
       print(f"for max laef nodes : {max_leaf_nodes:<5} mae = {mae}")   

    print("\nRandom Forest Regressor with limided features")
    for max_leaf_nodes in [5, 50, 500, 1000, 2000, 3000]:
       mae = getMea_RandomForestRegressor(max_leaf_nodes, train_X_lim, val_X_lim, train_y_lim, val_y_lim)
       print(f"for max laef nodes : {max_leaf_nodes:<5} mae = {mae}")

    print("\nRandom Forest Regressor with full features")
    for max_leaf_nodes in [5, 50, 500, 1000, 2000, 3000]:
       mae = getMea_RandomForestRegressor(max_leaf_nodes, train_X_full, val_X_full, train_y_full, val_y_full)
       print(f"for max laef nodes : {max_leaf_nodes:<5} mae = {mae}")

except FileNotFoundError:
    print("file not found")