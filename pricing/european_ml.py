from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def prepare_features(df):
    # Features: underlying_price, strike, T, r, sigma
    return df[["underlying_price", "strike", "T", "r", "sigma"]].values

def train_predict_linear_regression(df):
    features = prepare_features(df)
    # Separate call and put prices for training
    call_target = df["BS_call"].values  # Using BS call prices as target baseline
    put_target = df["BS_put"].values

    X_train, X_test, y_call_train, y_call_test, y_put_train, y_put_test = train_test_split(
        features, call_target, put_target, test_size=0.3, random_state=42
    )

    # Train Linear Regression for calls
    lr_call = LinearRegression().fit(X_train, y_call_train)
    # Train Linear Regression for puts
    lr_put = LinearRegression().fit(X_train, y_put_train)

    # Predict on entire dataset
    call_pred = lr_call.predict(features)
    put_pred = lr_put.predict(features)

    return call_pred, put_pred

def train_predict_random_forest(df):
    features = prepare_features(df)
    call_target = df["BS_call"].values
    put_target = df["BS_put"].values

    X_train, X_test, y_call_train, y_call_test, y_put_train, y_put_test = train_test_split(
        features, call_target, put_target, test_size=0.3, random_state=42
    )

    rf_call = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_call_train)
    rf_put = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_put_train)

    call_pred = rf_call.predict(features)
    put_pred = rf_put.predict(features)

    return call_pred, put_pred

def train_predict_ann(df):
    features = prepare_features(df)
    call_target = df["BS_call"].values
    put_target = df["BS_put"].values

    X_train, X_test, y_call_train, y_call_test, y_put_train, y_put_test = train_test_split(
        features, call_target, put_target, test_size=0.3, random_state=42
    )

    ann_call = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1000, random_state=42).fit(X_train, y_call_train)
    ann_put = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1000, random_state=42).fit(X_train, y_put_train)

    call_pred = ann_call.predict(features)
    put_pred = ann_put.predict(features)

    return call_pred, put_pred
