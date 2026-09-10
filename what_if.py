import pandas as pd


def simulate_health_change(
    model,
    sleep,
    hydration,
    stress,
    activity,
    caffeine
):
    """
    Run the prediction model using hypothetical
    health values.
    """

    health_data = {
        "sleep_hours": sleep,
        "hydration_liters": hydration,
        "stress": stress,
        "activity_steps": activity,
        "caffeine": caffeine
    }

    # Keep the same feature names used during training
    input_data = pd.DataFrame(
        [health_data]
    )

    probability = model.predict_proba(
        input_data
    )[0][1]

    return probability