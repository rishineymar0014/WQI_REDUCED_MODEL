"""
Research Question 2:
Can reduced-parameter monitoring provide reliable WQI prediction
for data-scarce river systems?

Pipeline: load data -> compute WQI (full parameters) -> train models on
reduced parameter sets -> test under data scarcity & on unseen stations.
Usage:  python wqi_reduced_model.py [path/to/merged_cpcb.csv]
CSV needs columns: station, temp, ph, do, bod, conductivity, nitrate, total_coliform
(rename your CPCB columns in COLUMN_MAP below).
"""
import sys, itertools, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ---------- 1. DATA ----------
COLUMN_MAP = {  # CPCB column name -> standard name (edit to match your CSVs)
    "Temperature": "temp", "pH": "ph", "Dissolved Oxygen": "do", "BOD": "bod",
    "Conductivity": "conductivity", "Nitrate": "nitrate",
    "Total Coliform": "total_coliform", "Station Name": "station",
}
PARAMS = ["temp", "ph", "do", "bod", "conductivity", "nitrate", "total_coliform"]

def synthetic_data(n_st=25, n_per=120):
    """Stand-in data with realistic ranges/correlations (use ONLY to test the pipeline)."""
    rows = []
    for s in range(n_st):
        pollution = rng.uniform(0, 1)
        for _ in range(n_per):
            p = np.clip(pollution + rng.normal(0, .15), 0, 1)
            temp = rng.normal(25, 3)
            do = np.clip(8.5 - 5.5*p - 0.1*(temp-25) + rng.normal(0, .6), 0.5, 10)
            bod = np.clip(1.2 + 14*p + rng.normal(0, 1), 0.5, 30)
            rows.append(dict(station=f"S{s}", temp=temp,
                ph=np.clip(rng.normal(7.6 + 0.3*(1-p), .35), 6, 9),
                do=do, bod=bod,
                conductivity=np.clip(200 + 900*p + rng.normal(0, 80), 50, 2500),
                nitrate=np.clip(0.5 + 9*p + rng.normal(0, 1), 0.05, 30),
                total_coliform=np.clip(np.exp(3.5 + 5*p + rng.normal(0, .8)), 5, 1e5)))
    return pd.DataFrame(rows)

def load_data(path=None):
    if path:
        df = pd.read_csv(path).rename(columns=COLUMN_MAP)
        df = df[[c for c in ["station"] + PARAMS if c in df.columns]]
        for c in PARAMS: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=PARAMS)
        print("Loaded real data:", df.shape); return df
    print("No CSV given -> using SYNTHETIC data (pipeline test only)")
    return synthetic_data()

# ---------- 2. WQI (weighted arithmetic index, BIS/CPCB-style standards) ----------
STD  = dict(ph=8.5, do=5.0, bod=3.0, conductivity=300.0, nitrate=45.0, total_coliform=500.0, temp=30.0)
IDEAL= dict(ph=7.0, do=14.6, bod=0.0, conductivity=0.0, nitrate=0.0, total_coliform=0.0, temp=0.0)
K = 1 / sum(1/v for v in STD.values())
W = {p: K / STD[p] for p in STD}

def compute_wqi(df):
    wqi = 0
    for p in STD:
        q = 100 * (df[p] - IDEAL[p]) / (STD[p] - IDEAL[p])
        q = q.clip(0, 200)  # cap sub-index so one extreme value (e.g. coliform) can't dominate
        wqi = wqi + W[p] * q
    return wqi / sum(W.values())

def wqi_class(w):
    return pd.cut(w, [-np.inf, 25, 50, 75, 100, np.inf],
                  labels=["Excellent", "Good", "Poor", "Very poor", "Unfit"])

# ---------- 3. EVALUATION HELPERS ----------
def rf(): return RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)

def score(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr); p = model.predict(Xte)
    acc = (wqi_class(p).astype(str) == wqi_class(yte).astype(str)).mean()
    return r2_score(yte, p), np.sqrt(mean_squared_error(yte, p)), acc

def main(path=None):
    df = load_data(path)
    df["wqi"] = compute_wqi(df)
    df["wqi_class"] = wqi_class(df["wqi"])
    print(df["wqi_class"].value_counts(), "\n")

    # --- A. Which small parameter sets work best? (exhaustive search, 3-5 params)
    tr, te = train_test_split(df, test_size=.25, random_state=1)
    rows = []
    for k in (7, 5, 4, 3, 2):
        for combo in itertools.combinations(PARAMS, k):
            r2, rmse, acc = score(LinearRegression(), tr[list(combo)], tr.wqi, te[list(combo)], te.wqi)
            rows.append((k, combo, r2, rmse, acc))
    res = pd.DataFrame(rows, columns=["k", "params", "r2_lr", "rmse_lr", "class_acc_lr"])
    best = res.sort_values("r2_lr", ascending=False).groupby("k").head(1).sort_values("k", ascending=False)
    print("Best parameter subset per size (Linear Regression):\n", best.to_string(index=False), "\n")
    best_sets = {k: list(p) for k, p in zip(best.k, best.params)}

    # --- B. Full vs reduced, Linear Regression vs Random Forest
    out = []
    for k, cols in best_sets.items():
        for name, m in (("LinearReg", LinearRegression()), ("RandomForest", rf())):
            r2, rmse, acc = score(m, tr[cols], tr.wqi, te[cols], te.wqi)
            out.append(dict(n_params=k, model=name, params=",".join(cols), R2=r2, RMSE=rmse, class_acc=acc))
    comp = pd.DataFrame(out); print(comp.round(3).to_string(index=False), "\n")

    # --- C. Data scarcity: shrink training data
    fracs = [.02, .05, .1, .2, .4, .7, 1.0]; scar = {}
    for k in (7, 4, 3):
        cols = best_sets[k]; scar[k] = []
        for f in fracs:
            vals = []
            for seed in range(5):
                sub = tr.sample(frac=f, random_state=seed) if f < 1 else tr
                if len(sub) < 10: sub = tr.sample(10, random_state=seed)
                vals.append(score(rf(), sub[cols], sub.wqi, te[cols], te.wqi)[0])
            scar[k].append(np.mean(vals))

    # --- D. Unseen stations (leave-stations-out) = transfer to new river
    cv = {}
    if "station" in df:
        gkf = GroupKFold(5)
        for k in (7, 4, 3):
            cols = best_sets[k]; r = []
            for a, b in gkf.split(df, groups=df.station):
                r.append(score(rf(), df.iloc[a][cols], df.iloc[a].wqi, df.iloc[b][cols], df.iloc[b].wqi)[0])
            cv[k] = np.mean(r)
        print("Leave-stations-out R2:", {k: round(v, 3) for k, v in cv.items()}, "\n")

    # --- E. Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, v in scar.items(): ax[0].plot([f*len(tr) for f in fracs], v, marker="o", label=f"{k} params")
    ax[0].set_xscale("log"); ax[0].set_xlabel("training samples"); ax[0].set_ylabel("R²")
    ax[0].set_title("Data-scarcity test"); ax[0].legend(); ax[0].grid(alpha=.3)
    piv = comp[comp.model == "RandomForest"]
    ax[1].bar(piv.n_params.astype(str), piv.R2); ax[1].set_ylim(min(.5, piv.R2.min()-.05), 1)
    ax[1].set_xlabel("number of parameters"); ax[1].set_ylabel("R² (RF)"); ax[1].set_title("Reduced vs full")
    plt.tight_layout(); plt.savefig("wqi_reduced_results.png", dpi=140)
    comp.to_csv("wqi_model_comparison.csv", index=False)

    # --- F. Verdict
    full, red = piv[piv.n_params == 7].R2.iloc[0], piv[piv.n_params == 3].R2.iloc[0]
    print(f"VERDICT: full(7) R2={full:.3f} | reduced(3: {best_sets[3]}) R2={red:.3f} | drop={full-red:.3f}")
    print("Reliable" if full - red < .05 else "Noticeable loss", "with 3 parameters.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
