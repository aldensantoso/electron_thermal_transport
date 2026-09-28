import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

def train_pipeline(df, *args, **kwargs):
    """
    Backward-compatible wrapper.
    Supports:
      - train_pipeline(df, feature_cols, target_col)
      - train_pipeline(df, X_cols, Y_wave_cols, Y_target_col)
    """
    if len(args) == 2:
        feature_cols, target_col = args
        y_wave_cols = None
    elif len(args) == 3:
        feature_cols, y_wave_cols, target_col = args
    else:
        raise TypeError(
            "train_pipeline() accepts either "
            "(df, feature_cols, target_col) or "
            "(df, X_cols, Y_wave_cols, Y_target_col)"
        )

    # normalize old API to new API
    X_cols = list(feature_cols)
    y_cols = list(y_wave_cols) if y_wave_cols is not None else [target_col]

    # example: use the existing training logic below
    # replace this block with your real model code
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X = df[X_cols].copy()
    y = df[y_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model1 = RandomForestRegressor(random_state=42, n_estimators=200)
    model1.fit(X_train_scaled, y_train)

    model2 = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=200))
    model2.fit(X_train_scaled, y_train)

    return model1, model2, scaler, scaler

def optimize_operating_space(model1, model2=None, scale1=None, scale2=None, X_cols=None):
    """
    Backward-compatible wrapper for older notebook code.
    """
    if X_cols is None:
        X_cols = []

    return {
        "model1": model1,
        "model2": model2,
        "scale1": scale1,
        "scale2": scale2,
        "X_cols": X_cols,
    }


def evaluate_and_report(model1, model2=None, X_test=None, y_test=None, feature_names=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error

    results = {}

    if X_test is None or y_test is None:
        return {"model1": model1, "model2": model2, "feature_names": feature_names}

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    pred1 = model1.predict(X_test)
    rmse1 = np.sqrt(np.mean((y_test - pred1) ** 2))
    results["model1"] = {
        "r2": r2_score(y_test, pred1),
        "rmse": rmse1,
        "predictions": pred1,
    }

    if model2 is not None:
        pred2 = model2.predict(X_test)
        rmse2 = np.sqrt(np.mean((y_test - pred2) ** 2))
        results["model2"] = {
            "r2": r2_score(y_test, pred2),
            "rmse": rmse2,
            "predictions": pred2,
        }

    if feature_names is not None and hasattr(model1, "feature_importances_"):
        importances = pd.Series(model1.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        importances.plot(kind="bar")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

        results["feature_importance"] = importances

    return results