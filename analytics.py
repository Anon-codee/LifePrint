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

def headache_sleep_pattern(df):
    """
    Find how often headaches occurred after
    below-average sleep.
    """

    headache_days = df[df["headache"] == 1]

    if len(headache_days) == 0:
        return {
            "total_headaches": 0,
            "below_average_sleep": 0,
            "percentage": 0
        }

    personal_sleep_baseline = df["sleep_hours"].mean()

    below_average = (
        headache_days["sleep_hours"]
        < personal_sleep_baseline
    )

    count = int(below_average.sum())

    percentage = (
        count / len(headache_days)
    ) * 100

    return {
        "total_headaches": len(headache_days),
        "below_average_sleep": count,
        "percentage": round(percentage, 1)
    }

def discover_headache_relationships(df):
    """
    Discover relationships between health variables
    and headache events using correlation.
    """

    features = [
        "sleep_hours",
        "hydration_liters",
        "stress",
        "activity_steps",
        "caffeine"
    ]

    relationships = []

    for feature in features:

        correlation = df[feature].corr(
            df["headache"]
        )

        relationships.append({
            "feature": feature,
            "correlation": round(float(correlation), 3),
            "strength": round(abs(float(correlation)), 3)
        })

    # Strongest relationships first
    relationships.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return relationships

def relationship_label(strength):

    if strength >= 0.5:
        return "Strong"

    elif strength >= 0.3:
        return "Moderate"

    elif strength >= 0.15:
        return "Weak"

    else:
        return "Very Weak"