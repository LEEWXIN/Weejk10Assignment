"""
Streamlit app: loads the already-fitted HDBSCAN pipeline (NOT retrained here)
and lets a user assign a new / hypothetical customer to a segment in real time.

Run:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation Explorer", layout="wide")

PIPELINE_PATH = "outputs/model/clustering_pipeline.joblib"
DATA_PATH = "outputs/customer_features_clustered.csv"

# Human-readable names + narrative for each cluster id, based on the
# statistical profiling done in the notebook (Cell 21 groupby median table).
CLUSTER_INFO = {
    -1: {
        "name": "Heterogeneous / edge customers",
        "desc": "Doesn't match either archetype tightly -- moderate recency, "
                "frequency and spend across the board. Not bad data, just "
                "behaviourally in-between. Worth manual review rather than "
                "automated targeting.",
    },
    0: {
        "name": "At-risk, high-return sleeper customers",
        "desc": "Bought only once, ~8 months ago, and returned about half of "
                "what they bought. Small group (30 customers) but a strong "
                "churn/product-fit warning signal worth investigating.",
    },
    1: {
        "name": "Core active customers",
        "desc": "The main customer base -- recent purchases (~40 days), "
                "essentially no returns, moderate but steady spend. This is "
                "the group to retain with loyalty incentives.",
    },
}

# Must match the `features` list from your notebook, in the same order
FEATURES = [
    "Recency", "Frequency", "Monetary", "AvgUnitPrice",
    "DistinctProducts", "TotalItems", "AvgBasketValue", "ReturnRate",
]
# Must match the `log_features` list from Cell 14
LOG_FEATURES = [
    "Frequency", "Monetary", "AvgUnitPrice",
    "DistinctProducts", "TotalItems", "AvgBasketValue",
]


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


def transform_new_point(raw_values: dict, scaler):
    x = pd.DataFrame([raw_values])[FEATURES]
    for col in LOG_FEATURES:
        x[col] = np.log1p(x[col])
    return scaler.transform(x)


def main():
    pipeline = load_pipeline()
    ref_df = load_reference_data()
    scaler = pipeline["scaler"]
    clusterer = pipeline["clusterer"]
    reducer = pipeline["umap_reducer"]

    st.title("From Clusters to Product: Customer Segmentation Explorer")
    st.caption(
        "Dataset: UCI Online Retail (Chen, 2015, CC BY 4.0) Â· "
        "Method: HDBSCAN on RFM + behavioural features, visualized with UMAP"
    )

    left, right = st.columns([1.1, 1])

    # ---------------- LEFT: existing cluster map ----------------
    with left:
        st.subheader("Existing customer segments (UMAP projection)")
        fig, ax = plt.subplots(figsize=(7, 6))
        cluster_ids = sorted(ref_df["cluster"].unique())
        palette = {cid: color for cid, color in zip(
            cluster_ids, sns.color_palette("tab10", len(cluster_ids))
        )}
        for cid, sub in ref_df.groupby("cluster"):
            color = "lightgray" if cid == -1 else palette[cid]
            label = CLUSTER_INFO.get(cid, {}).get("name", f"Cluster {cid}")
            ax.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.55, color=color, label=label)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        st.pyplot(fig)

        with st.expander("Cluster profiles & internal validation metrics"):
            profile = ref_df.groupby("cluster")[FEATURES].median().round(2)
            profile["count"] = ref_df.groupby("cluster").size()
            st.dataframe(profile)

    # ---------------- RIGHT: live inference on a new customer ----------------
    with right:
        st.subheader("Assign a new customer to a segment")
        st.write("Enter a customer's behaviour and the saved pipeline will classify them live.")

        c1, c2 = st.columns(2)
        with c1:
            recency = st.number_input("Recency (days since last purchase)", 0, 400, 30)
            frequency = st.number_input("Frequency (distinct orders)", 1, 300, 4)
            monetary = st.number_input("Monetary (total spend, GBP)", 0.0, 300000.0, 800.0)
            avg_unit_price = st.number_input("Average unit price (GBP)", 0.0, 2500.0, 3.5)
        with c2:
            distinct_products = st.number_input("Distinct products bought", 1, 2000, 40)
            total_items = st.number_input("Total items bought", 1, 250000, 500)
            avg_basket = st.number_input("Average basket value (GBP)", 0.0, 100000.0, 250.0)
            return_rate = st.slider("Return rate (0-1)", 0.0, 1.0, 0.0, 0.05)

        if st.button("Classify this customer", type="primary"):
            raw = {
                "Recency": recency, "Frequency": frequency, "Monetary": monetary,
                "AvgUnitPrice": avg_unit_price, "DistinctProducts": distinct_products,
                "TotalItems": total_items, "AvgBasketValue": avg_basket,
                "ReturnRate": return_rate,
            }
            x_scaled = transform_new_point(raw, scaler)
            label, strength = hdbscan.approximate_predict(clusterer, x_scaled)
            label = int(label[0])
            info = CLUSTER_INFO.get(label, {"name": f"Cluster {label}", "desc": "No description available."})

            st.markdown(f"### Result: **{info['name']}**")
            st.write(info["desc"])
            st.caption(f"Membership strength: {strength[0]:.2f} (HDBSCAN confidence, 0-1; low = borderline case)")

            # Project the new point onto the existing UMAP space and show where it lands
            new_xy = reducer.transform(x_scaled)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            for cid, sub in ref_df.groupby("cluster"):
                color = "lightgray" if cid == -1 else palette[cid]
                ax2.scatter(sub["umap_x"], sub["umap_y"], s=8, alpha=0.35, color=color)
            ax2.scatter(new_xy[0, 0], new_xy[0, 1], s=250, marker="*", color="black", label="New customer")
            ax2.legend()
            ax2.set_xlabel("UMAP-1")
            ax2.set_ylabel("UMAP-2")
            st.pyplot(fig2)


if __name__ == "__main__":
    main()