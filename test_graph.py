import pandas as pd

from analytics import discover_headache_relationships

from graph import build_health_graph


# Load health history
df = pd.read_csv(
    "data/health_data.csv"
)


# Discover relationships
relationships = discover_headache_relationships(
    df
)


# Build Personal Health Graph
graph = build_health_graph(
    relationships
)


print("\n========== LIFEPRINT HEALTH GRAPH ==========\n")

print("NODES:")

for node in graph.nodes:

    print(
        f"  • {node}"
    )


print("\nRELATIONSHIPS:")

for source, target, data in graph.edges(data=True):

    print(
        f"  {source} → {target} | "
        f"strength: {data['strength']:.3f} | "
        f"direction: {data['direction']}"
    )