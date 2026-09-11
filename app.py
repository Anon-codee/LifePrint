import streamlit as st
import pandas as pd
import networkx as nx

from model import train_model, predict_headache, explain_prediction
from what_if import simulate_health_change
from analytics import discover_headache_relationships


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LifePrint",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/health_data.csv")


df = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def get_model(data):
    return train_model(data)


model = get_model(df)


# ============================================================
# PERSONAL BASELINE
# ============================================================

health_features = [
    "sleep_hours",
    "hydration_liters",
    "stress",
    "activity_steps",
    "caffeine"
]

baseline = {
    feature: df[feature].mean()
    for feature in health_features
}


# ============================================================
# CURRENT DAY
# ============================================================

today = {
    "sleep_hours": 5.2,
    "hydration_liters": 1.4,
    "stress": 8,
    "activity_steps": 4000,
    "caffeine": 3
}


# ============================================================
# PREDICTION
# ============================================================

risk = predict_headache(
    model,
    today
)


explanations = explain_prediction(
    model,
    today
)


# ============================================================
# PATTERN
# ============================================================

headache_days = df[df["headache"] == 1]

below_sleep_headaches = headache_days[
    headache_days["sleep_hours"] < baseline["sleep_hours"]
]

if len(headache_days) > 0:
    sleep_pattern_percentage = (
        len(below_sleep_headaches)
        / len(headache_days)
        * 100
    )
else:
    sleep_pattern_percentage = 0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧬 LifePrint")

st.sidebar.markdown(
    """
    ### Your Health. Your Story.

    LifePrint learns recurring patterns
    from your longitudinal health history.
    """
)

section = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Health Timeline",
        "Personal Health Graph",
        "What-If Simulator"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("LIFEPRINT")

st.subheader(
    "A Personalized Health Digital Twin"
)

st.caption(
    "From tracking health parameters → to learning the individual."
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if section == "Dashboard":

    st.header("Today's Health")

    cols = st.columns(5)

    labels = {
        "sleep_hours": "Sleep",
        "hydration_liters": "Hydration",
        "stress": "Stress",
        "activity_steps": "Activity",
        "caffeine": "Caffeine"
    }

    units = {
        "sleep_hours": "hrs",
        "hydration_liters": "L",
        "stress": "/10",
        "activity_steps": "steps",
        "caffeine": ""
    }

    for col, feature in zip(cols, health_features):

        current = today[feature]
        base = baseline[feature]

        if base != 0:
            deviation = (
                (current - base)
                / base
                * 100
            )
        else:
            deviation = 0

        col.metric(
            labels[feature],
            f"{current:.1f} {units[feature]}",
            f"{deviation:+.1f}% vs baseline"
        )

    st.divider()

    # Prediction section
    st.header("🔮 Headache Likelihood")

    prediction_col, explanation_col = st.columns(
        [1, 2]
    )

    with prediction_col:

        st.metric(
            "Estimated likelihood",
            f"{risk * 100:.1f}%"
        )

        if risk >= 0.7:
            st.warning("Higher model-estimated likelihood")
        elif risk >= 0.4:
            st.info("Moderate model-estimated likelihood")
        else:
            st.success("Lower model-estimated likelihood")

    with explanation_col:

        st.subheader("Why this prediction?")

        for item in explanations[:3]:

            feature_name = item["feature"].replace(
                "_",
                " "
            ).title()

            st.write(
                f"**{feature_name}** — "
                f"model importance: "
                f"{item['importance']:.3f}"
            )

    st.divider()

    # Personal learning
    st.header("🧠 What LifePrint Learned")

    st.info(
        f"{sleep_pattern_percentage:.1f}% of your "
        f"recorded headaches occurred after "
        f"below-average sleep."
    )

    st.caption(
        f"Based on {len(headache_days)} recorded "
        f"headache events in the dataset."
    )

    st.divider()

    st.header("Your Personal Baseline")

    baseline_cols = st.columns(5)

    for col, feature in zip(
        baseline_cols,
        health_features
    ):

        col.metric(
            labels[feature],
            f"{baseline[feature]:.2f}"
        )


# ============================================================
# TIMELINE
# ============================================================

elif section == "Health Timeline":

    st.header("📈 Your Health Timeline")

    st.write(
        "LifePrint uses longitudinal data to understand "
        "how your health changes over time."
    )

    chart_data = df.copy()

    chart_data["date"] = pd.to_datetime(
        chart_data["date"]
    )

    chart_data = chart_data.set_index("date")

    st.subheader("Sleep")

    st.line_chart(
        chart_data["sleep_hours"]
    )

    st.subheader("Hydration")

    st.line_chart(
        chart_data["hydration_liters"]
    )

    st.subheader("Stress")

    st.line_chart(
        chart_data["stress"]
    )

    st.subheader("Activity")

    st.line_chart(
        chart_data["activity_steps"]
    )

    st.subheader("Caffeine")

    st.line_chart(
        chart_data["caffeine"]
    )

    st.subheader("Headache Events")

    st.bar_chart(
        chart_data["headache"]
    )


# ============================================================
# HEALTH GRAPH
# ============================================================

elif section == "Personal Health Graph":

    st.header("🧠 Personal Health Graph")

    st.write(
        "Relationships discovered from your "
        "longitudinal health history."
    )

    relationships = discover_headache_relationships(
        df
    )

    graph = nx.DiGraph()

    graph.add_node(
        "Headache"
    )

    names = {
        "sleep_hours": "Sleep",
        "hydration_liters": "Hydration",
        "stress": "Stress",
        "activity_steps": "Activity",
        "caffeine": "Caffeine"
    }

    for relationship in relationships:

        feature = relationship["feature"]

        readable = names.get(
            feature,
            feature
        )

        graph.add_node(
            readable
        )

        graph.add_edge(
            readable,
            "Headache",
            strength=relationship["strength"],
            direction=(
                "positive"
                if relationship["correlation"] > 0
                else "negative"
            )
        )

    st.subheader("Learned Relationships")

    for source, target, data in graph.edges(
        data=True
    ):

        symbol = (
            "+"
            if data["direction"] == "positive"
            else "−"
        )

        st.write(
            f"**{source} → {target}**  "
            f"{symbol} "
            f"strength: {data['strength']:.3f}"
        )

    st.divider()

    st.subheader("Graph Structure")

    graph_data = pd.DataFrame(
        [
            {
                "Factor": source,
                "Relationship": f"{data['direction']} association",
                "Strength": data["strength"]
            }
            for source, target, data
            in graph.edges(data=True)
        ]
    )

    st.dataframe(
        graph_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Relationships represent statistical associations "
        "in the prototype dataset, not medical causation."
    )


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

elif section == "What-If Simulator":

    st.header("🔮 What-If Health Simulator")

    st.write(
        "Change your hypothetical health conditions "
        "and see how the prediction model responds."
    )

    st.warning(
        "This is a model simulation, not a medical "
        "prediction or treatment recommendation."
    )

    col1, col2 = st.columns(2)

    with col1:

        sleep = st.slider(
            "Sleep (hours)",
            min_value=3.0,
            max_value=10.0,
            value=5.2,
            step=0.1
        )

        hydration = st.slider(
            "Hydration (liters)",
            min_value=0.5,
            max_value=4.0,
            value=1.4,
            step=0.1
        )

        stress = st.slider(
            "Stress",
            min_value=0,
            max_value=10,
            value=8
        )

    with col2:

        activity = st.slider(
            "Activity (steps)",
            min_value=1000,
            max_value=15000,
            value=4000,
            step=500
        )

        caffeine = st.slider(
            "Caffeine",
            min_value=0,
            max_value=6,
            value=3,
            step=1
        )

    simulated_risk = simulate_health_change(
        model,
        sleep=sleep,
        hydration=hydration,
        stress=stress,
        activity=activity,
        caffeine=caffeine
    )

    st.divider()

    current_col, simulated_col = st.columns(2)

    with current_col:

        st.metric(
            "Current estimate",
            f"{risk * 100:.1f}%"
        )

    with simulated_col:

        st.metric(
            "Simulated estimate",
            f"{simulated_risk * 100:.1f}%",
            f"{(simulated_risk - risk) * 100:+.1f} points"
        )

    st.divider()

    st.subheader("Simulation Summary")

    st.write(
        f"""
        With the selected hypothetical conditions:

        - Sleep: **{sleep:.1f} hours**
        - Hydration: **{hydration:.1f} L**
        - Stress: **{stress}/10**
        - Activity: **{activity:,} steps**
        - Caffeine: **{caffeine}**
        """
    )

    st.info(
        "LifePrint recalculates the model output using "
        "the hypothetical values. This demonstrates "
        "model sensitivity, not proof of causality."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LifePrint Prototype • Research demonstration • "
    "Not intended for medical diagnosis or treatment."
)