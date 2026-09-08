import pandas as pd
import numpy as np

np.random.seed(42)

# 120 days of personal health history
days = 120

dates = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=days
)

# Generate realistic health variables
sleep = np.clip(
    np.random.normal(7, 1.1, days),
    4,
    9
)

hydration = np.clip(
    np.random.normal(2.0, 0.5, days),
    0.8,
    3.5
)

stress = np.clip(
    np.random.normal(5, 2, days),
    1,
    10
)

activity = np.clip(
    np.random.normal(7000, 1800, days),
    1000,
    12000
)

caffeine = np.clip(
    np.random.normal(2, 1, days),
    0,
    5
)

# ------------------------------------------------
# Create a PERSONAL relationship between variables
# and headache occurrence.
# ------------------------------------------------

headache_score = (
    (7 - sleep) * 1.2
    + (stress - 5) * 0.7
    + (2 - hydration) * 1.0
    + caffeine * 0.25
    + np.random.normal(0, 1, days)
)

headache_probability = 1 / (
    1 + np.exp(-headache_score)
)

headache = (
    np.random.random(days)
    < headache_probability * 0.35
).astype(int)

# Create dataset
df = pd.DataFrame({
    "date": dates,
    "sleep_hours": np.round(sleep, 2),
    "hydration_liters": np.round(hydration, 2),
    "stress": np.round(stress, 1),
    "activity_steps": activity.astype(int),
    "caffeine": np.round(caffeine, 1),
    "headache": headache
})

# Save
df.to_csv(
    "data/health_data.csv",
    index=False
)

print("\nLifePrint dataset created!")
print(f"Days recorded: {len(df)}")
print(f"Headache events: {df['headache'].sum()}")

print("\nFirst 5 records:")
print(df.head())