import pandas as pd

from analytics import (
    discover_headache_relationships,
    relationship_label
)


df = pd.read_csv(
    "data/health_data.csv"
)

relationships = discover_headache_relationships(df)


print("\n========== PERSONAL HEALTH GRAPH ==========\n")

for relationship in relationships:

    feature = relationship["feature"]
    correlation = relationship["correlation"]
    strength = relationship["strength"]

    label = relationship_label(strength)

    print(
        f"{feature:20} "
        f"{correlation:+.3f} "
        f"{label}"
    )