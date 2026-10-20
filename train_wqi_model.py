"""
Reduced-parameter WQI prediction model (Research Question 2).
Trains Random Forest models on 3, 4 and 7 parameters, saves them,
and gives a predict() function.  Run:  python train_wqi_model.py
"""
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from wqi_reduced_model import synthetic_data, compute_wqi, wqi_class   # data + WQI formula

FEATURE_SETS = {
    "3_params": ["ph", "bod", "total_coliform"],
    "4_params": ["ph", "do", "bod", "total_coliform"],
    "7_params": ["temp", "ph", "do", "bod", "conductivity", "nitrate", "total_coliform"],
}

df = synthetic_data(n_st=40, n_per=150)
df["wqi"] = compute_wqi(df)
train, test = train_test_split(df, test_size=0.25, random_state=1)

models = {}
for name, cols in FEATURE_SETS.items():
    m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)
    m.fit(train[cols], train.wqi)
    p = m.predict(test[cols])
    print(f"{name}: R2={r2_score(test.wqi, p):.3f}  RMSE={np.sqrt(mean_squared_error(test.wqi, p)):.2f}")
    models[name] = (m, cols)

joblib.dump(models, "wqi_models.joblib")

def predict_wqi(**readings):
    """Pick the smallest model whose parameters you have. e.g. predict_wqi(ph=7.4, bod=5, total_coliform=900)"""
    for name in ("3_params", "4_params", "7_params"):
        m, cols = models[name]
        if set(cols) <= set(readings):
            w = float(m.predict(pd.DataFrame([{c: readings[c] for c in cols}]))[0])
            return {"model": name, "WQI": round(w, 1), "class": str(wqi_class(pd.Series([w]))[0])}
    raise ValueError("Need at least ph, bod and total_coliform")

if __name__ == "__main__":
    print(predict_wqi(ph=7.4, bod=5.0, total_coliform=900))
