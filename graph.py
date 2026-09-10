import networkx as nx


def build_health_graph(relationships):
    """
    Build a Personal Health Graph from the
    relationships discovered from the user's data.
    """

    graph = nx.DiGraph()

    # Target health event
    target = "Headache"

    # Add target node
    graph.add_node(
        target,
        type="event"
    )

    # Add discovered relationships
    for relationship in relationships:

        feature = relationship["feature"]
        correlation = relationship["correlation"]
        strength = relationship["strength"]

        # Convert technical column names
        # into readable names
        names = {
            "sleep_hours": "Sleep",
            "hydration_liters": "Hydration",
            "stress": "Stress",
            "activity_steps": "Activity",
            "caffeine": "Caffeine"
        }

        readable_name = names.get(
            feature,
            feature
        )

        graph.add_node(
            readable_name,
            type="health_factor"
        )

        # Direction based on correlation
        if correlation < 0:
            direction = "negative"
        else:
            direction = "positive"

        graph.add_edge(
            readable_name,
            target,
            strength=strength,
            direction=direction
        )

    return graph