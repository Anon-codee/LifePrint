import pandas as pd

from model import (
    train_model,
    predict_headache
)

from what_if import (
    simulate_health_change
)


# Load historical data
df = pd.read_csv(
    "data/health_data.csv"
)


# Train model
model = train_model(df)


# Current health
current = {
    "sleep_hours": 5.2,
    "hydration_liters": 1.4,
    "stress": 8,
    "activity_steps": 4000,
    "caffeine": 3
}


# Current prediction
current_risk = predict_headache(
    model,
    current
)


# Hypothetical improved conditions
simulated_risk = simulate_health_change(
    model,
    sleep=8.0,
    hydration=2.5,
    stress=3,
    activity=7000,
    caffeine=1
)


print("\n========== WHAT-IF SIMULATION ==========\n")

print(
    f"Current headache likelihood: "
    f"{current_risk * 100:.1f}%"
)

print(
    f"Simulated headache likelihood: "
    f"{simulated_risk * 100:.1f}%"
)

change = (
    simulated_risk - current_risk
) * 100

print(
    f"\nChange in model estimate: "
    f"{change:+.1f} percentage points"
)