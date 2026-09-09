import pandas as pd

from model import (
    train_model,
    predict_headache
)


# Load historical data
df = pd.read_csv(
    "data/health_data.csv"
)


# Train LifePrint's model
model = train_model(df)


# Example current-day health data
today = {
    "sleep_hours": 5.2,
    "hydration_liters": 1.4,
    "stress": 8,
    "activity_steps": 4000,
    "caffeine": 3
}


# Generate prediction
risk = predict_headache(
    model,
    today
)


print("\n========== LIFEPRINT PREDICTION ==========\n")

print(
    f"Estimated headache likelihood: "
    f"{risk * 100:.1f}%"
)