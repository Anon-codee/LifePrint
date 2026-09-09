import pandas as pd

from model import (
    train_model,
    predict_headache,
    explain_prediction
)


# Load historical health data
df = pd.read_csv(
    "data/health_data.csv"
)


# Train LifePrint prediction model
model = train_model(df)


# Today's health data
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


# Generate explanation
explanations = explain_prediction(
    model,
    today
)


# Display prediction
print("\n========== LIFEPRINT PREDICTION ==========\n")

print(
    f"Estimated headache likelihood: "
    f"{risk * 100:.1f}%"
)


# Display explanation
print("\n========== WHY THIS PREDICTION? ==========\n")

for item in explanations:

    print(
        f"{item['feature']:20} "
        f"importance: {item['importance']:.3f}"
    )