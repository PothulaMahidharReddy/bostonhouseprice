# create_model.py  (put inside bostonhouseprice folder, run once)

import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor    # any model you like

# ------------------------------------------------------------------
# 1. Fetch the Boston Housing dataset manually ---------------------
# ------------------------------------------------------------------
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df   = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# raw_df rows alternate between feature halves -> stitch them together
data   = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT'
]
X = pd.DataFrame(data, columns=feature_names)
y = target

# ------------------------------------------------------------------
# 2. Fit scaler and model ------------------------------------------
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

# ------------------------------------------------------------------
# 3. Persist artefacts  --------------------------------------------
# ------------------------------------------------------------------
BASE_DIR = r"M:\endtoend"              # write here so Flask can find them
with open(os.path.join(BASE_DIR, "remodel.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "scaling.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("✅ 13‑feature remodel.pkl and scaling.pkl saved to:", BASE_DIR)
