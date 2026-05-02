"""Script to generate synthetic house price dataset."""
import numpy as np
import pandas as pd

from pathlib import Path

np.random.seed(42)
n = 300

size = np.random.randint(500, 4000, n)            # sq ft
rooms = np.random.randint(1, 8, n)                # number of rooms
location_score = np.round(np.random.uniform(1, 10, n), 1)  # 1–10 score

# Price formula with noise
price = (
    80 * size
    + 15000 * rooms
    + 12000 * location_score
    + np.random.normal(0, 20000, n)
    + 50000
)
price = np.round(price, -2)  # round to nearest $100

df = pd.DataFrame({
    "size": size,
    "rooms": rooms,
    "location_score": location_score,
    "price": price
})

df.to_csv(Path(__file__).parent / "house_prices.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df.describe())
