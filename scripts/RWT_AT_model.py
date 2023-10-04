import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import math
from pathlib import Path
import joblib
from datetime import date
warnings.filterwarnings("ignore")

def train_model(model_type, X_train, y_train, freq):# Training
    model = None
    if model_type == 0:
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_type == 1:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

    elif model_type == 2:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

    joblib.dump(model, f"userdata/model_{model_type}_"+freq+".joblib")
    return model

def bootstrapping(model_type, X_train, y_train):# Bootstrapping
    model = None
    if model_type == 0:
        model = LinearRegression()

    elif model_type == 1:
        model = SVR(kernel="rbf")

    elif model_type == 2:
        model = RandomForestRegressor()

    # Bootstrapping
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(X_train) - 1, len(X_train))
        if len(np.unique(y_train.iloc[indices])) < 2:
            continue
        model.fit(X_train.iloc[indices], y_train.iloc[indices])
    joblib.dump(model, f"userdata/model_{model_type}_bootstrapped.joblib")    
    return model


def train_wt_at(
    model_type=0, perf_type=0, y_file="data/water_temp.csv", x_file="data/air_temp.csv"
):  # model: 0->LR, 1->SVR, 2->RFR; perf: 0->RMSE, 1->MAE, 2->MSE
    """
    Takes input model type (linear regression, SVR, Random forest regressor), performance measure type (MSE, MAE, RMSE),
    paths to RWT and AT csvs (assuming same number of datapoints)

    Trains based on chosen model and evaluates selected error type. Writes the model to .joblib

    Returns dict with error for training and testing data.
    """
    #MONTHLY
    X = pd.read_csv(x_file)[["avg_air_temp"]]
    y = pd.read_csv(y_file)[["avg_water_temp"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(model_type, X_train, y_train, "monthly")
    # plot regression line and save as RWT_AT.png
    if os.path.exists('static/RWT_AT.png'):
        os.remove('static/RWT_AT.png')
    if model_type == 0: #linear regression
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.plot(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT.png')
    else: #SVR and RFR
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.scatter(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT.png')

    #DAILY
    X = pd.read_csv("data/air_temp_daily.csv")[["avg_air_temp"]]
    y = pd.read_csv("data/water_temp_daily.csv")[["avg_water_temp"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(model_type, X_train, y_train, "daily")
    # plot regression line and save as RWT_AT_daily.png
    if os.path.exists('static/RWT_AT_extrapolated.png'):
        os.remove('static/RWT_AT_extrapolated.png')
    if model_type == 0: #linear regression
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.plot(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT_extrapolated.png')
    else: #SVR and RFR
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.scatter(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT_extrapolated.png')

    # Bootstrapping using the monthly data
    #MONTHLY
    X = pd.read_csv(x_file)[["avg_air_temp"]]
    y = pd.read_csv(y_file)[["avg_water_temp"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = bootstrapping(model_type, X_train, y_train)
    # plot regression line and save as RWT_AT_bootstrapped.png
    if os.path.exists('static/RWT_AT_bootstrapped.png'):
        os.remove('static/RWT_AT_bootstrapped.png')
    if model_type == 0: #linear regression
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.plot(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT_bootstrapped.png')
    else: #SVR and RFR
        plt.figure(figsize=(20,10))
        plt.scatter(X_train, y_train, color = 'red', label = 'training data')
        plt.scatter(X_test, y_test, color = 'blue', label = 'testing data')
        plt.scatter(X_train, model.predict(X_train), color = 'black', label = 'regression line')
        plt.xlabel('Air Temperature (deg C)', fontsize='16', weight='bold')
        plt.ylabel('River Water Temperature (deg C)', fontsize='16', weight='bold')
        plt.xticks(fontsize='14')
        plt.yticks(fontsize='14')
        plt.legend(fontsize = '12')
        plt.title('Regression Plot', fontsize = '18')
        plt.savefig('static/RWT_AT_bootstrapped.png')

    # Evaluating train/test data only DAILY
    train_err = 0
    test_err = 0
    if model is not None and perf_type == 0:
        test_err = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        train_err = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))

    elif model is not None and perf_type == 1:
        test_err = mean_absolute_error(y_test, model.predict(X_test))
        train_err = mean_absolute_error(y_train, model.predict(X_train))

    elif model is not None and perf_type == 2:
        test_err = mean_squared_error(y_test, model.predict(X_test))
        train_err = mean_squared_error(y_train, model.predict(X_train))

    return {"train_error": train_err, "test_error": test_err}

def make_prediction(model_type, freq, output_file, air_temp_file):
    model_file = f"userdata/model_{model_type}_"+freq+".joblib"
    if not Path(model_file).exists():
        raise Exception("Model not found")

    model = joblib.load(model_file)
    X = pd.read_csv(air_temp_file)[["avg_air_temp"]]

    y_pred = model.predict(X)
    y_pnew = []
    # print(type(y_pred[0]))
    if isinstance(y_pred[0], np.ndarray): #for linear regression
        for x in y_pred:
            y_pnew.append(x[0])
        y_pred = y_pnew

    if freq=="daily":
        dates = pd.read_csv(air_temp_file)[["date"]]
        df = pd.DataFrame({"date": dates["date"], "avg_water_temp": y_pred})
    else:
        dates = pd.read_csv(air_temp_file)[["month"]]
        df = pd.DataFrame({"month": dates["month"], "avg_water_temp": y_pred})
    df.to_csv(output_file, index=False)

def predict_wt_at(
    model_type=0,
    output_file="static/predicted_wt_daily.csv",
    air_temp_file="data/air_temp_daily.csv",
):
    # Takes input type of model (requires this model to exist, trained already), output file path and air temperature file path.
    # Writes predicted RWT values to the csv
    # Returns nothing
    #DAILY
    make_prediction(model_type, "daily", output_file, air_temp_file) 

    
    #MONTHLY
    make_prediction(model_type, "monthly", "static/predicted_wt_monthly.csv", "data/air_temp.csv")


    #Bootstrapped
    make_prediction(model_type, "bootstrapped", "static/predicted_wt_bootstrapped.csv", "data/air_temp.csv")

    if os.path.exists('static/RWT_AT_timeseries.png'):
        os.remove('static/RWT_AT_timeseries.png')
    plot_rwt_timeseries(output_file) 

def plot_rwt_timeseries(predicted_wt_file, air_temp_file = 'data/air_temp_daily.csv', obs_wt_file = 'data/water_temp.csv'):
    pred_data = pd.read_csv(predicted_wt_file)
    obs_data = pd.read_csv(obs_wt_file)
    df =pd.read_csv(air_temp_file)

    #get observed data for every 15 days
    mon_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    date_list = []
    for i in range(len(obs_data)):
        mon_str = obs_data.month[i]
        mon = mon_str.split('-')[0]
        yr = mon_str.split('-')[1]
        date_list.append(date(int(yr)+2000,mon_short.index(mon)+1,15).strftime('%d/%m/%Y'))
    odf = obs_data.copy()
    odf['Date'] = date_list
    odf = odf[['month','Date','avg_water_temp']]

    #plot
    plt.figure(figsize=(20,10))
    x = df.date
    y = df.avg_air_temp
    plt.plot(x, y, color = 'lightgrey', label = 'air temp')
    plt.plot(x, pred_data.avg_water_temp, label = 'predicted wt')
    plt.scatter(odf.Date, odf.avg_water_temp,color='red', label = 'observed wt')
    plt.xlabel('Date', fontsize='16', weight='bold')
    plt.ylabel('Temperature (deg C)', fontsize='16', weight='bold')
    _ = plt.xticks(x[::120],rotation=45,  fontsize='14')
    _ = plt.yticks(fontsize='14')
    plt.legend(fontsize = '12')
    plt.title('Timeseries Plot', fontsize = '18')
    plt.savefig('static/RWT_AT_timeseries.png')

print(train_wt_at(0, 2))
predict_wt_at(0)
