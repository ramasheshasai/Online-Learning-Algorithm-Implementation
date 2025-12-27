import numpy as np
import pandas as pd
import os

# ==== CONFIG ====
ROWS = 50000          # Change to 100000 or 1,000,000 if needed
FEATURES = 5          # Number of feature columns
SAVE_PATH = "data/stream.csv"
# =================

np.random.seed(42)

# Create folder if not exists
os.makedirs("data", exist_ok=True)

# Generate Features
X = np.random.normal(0, 1, (ROWS, FEATURES))

# Create linear relationship + noise
weights = np.random.uniform(-2, 2, FEATURES)
linear_combination = np.dot(X, weights)

# Convert to probability
prob = 1 / (1 + np.exp(-linear_combination))

# Binary labels
y = (prob >= 0.5).astype(int)

# Create dataframe
cols = [f"feat{i+1}" for i in range(FEATURES)]
df = pd.DataFrame(X, columns=cols)
df["target"] = y

# Save CSV
df.to_csv(SAVE_PATH, index=False)

print(f"Generated dataset with {ROWS} rows & {FEATURES} features at {SAVE_PATH}")
