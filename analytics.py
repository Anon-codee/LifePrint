import pandas as pd


def calculate_baseline(df):
    """
    Calculate the individual's normal health values
    from their historical data.
    """

    baseline = {
        "sleep_hours": df["sleep_hours"].mean(),
        "hydration_liters": df["hydration_liters"].mean(),
        "stress": df["stress"].mean(),
        "activity_steps": df["activity_steps"].mean(),
        "caffeine": df["caffeine"].mean()
    }

    return baseline


def compare_to_baseline(current, baseline):
    """
    Compare today's health values against
    the individual's personal baseline.
    """

    comparison = {}

    for key in baseline:

        difference = (
            (current[key] - baseline[key])
            / baseline[key]
        ) * 100

        comparison[key] = round(difference, 1)

    return comparison


if __name__ == "__main__":

    # Load historical data
    df = pd.read_csv(
        "data/health_data.csv"
    )

    # Calculate baseline
    baseline = calculate_baseline(df)

    print("\n========== PERSONAL BASELINE ==========\n")

    print(
        f"Average Sleep: "
        f"{baseline['sleep_hours']:.2f} hours"
    )

    print(
        f"Average Hydration: "
        f"{baseline['hydration_liters']:.2f} L"
    )

    print(
        f"Average Stress: "
        f"{baseline['stress']:.2f}/10"
    )

    print(
        f"Average Activity: "
        f"{baseline['activity_steps']:.0f} steps"
    )

    print(
        f"Average Caffeine: "
        f"{baseline['caffeine']:.2f}"
    )