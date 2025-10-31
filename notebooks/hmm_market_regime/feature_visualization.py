import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from sklearn.preprocessing import StandardScaler

    pio.templates.default = "seaborn"
    return Path, StandardScaler, mo, pd, px


@app.cell
def _(Path, pd):
    features_path = Path("BTCUSDT_features_wfo_1.csv")

    features = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    features
    return (features,)


@app.cell
def _(features, mo):
    feature_selector = mo.ui.dropdown(
        options=features.columns,
        value=features.columns[0],
        label="Check original feature",
    )
    feature_selector
    return (feature_selector,)


@app.cell
def _(feature_selector, features, px):
    selected_feature = feature_selector.value

    fig_feature = px.line(
        x=features[selected_feature].index,
        y=features[selected_feature],
    )
    fig_feature.update_layout(title=selected_feature, width=800, height=500)
    fig_feature
    return


@app.cell
def _(StandardScaler, features, pd):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(
        features_scaled, index=features.index, columns=features.columns
    )
    features_scaled_df
    return (features_scaled_df,)


@app.cell
def _(features_scaled_df, mo):
    scaled_feature_selector = mo.ui.dropdown(
        options=features_scaled_df.columns,
        value=features_scaled_df.columns[0],
        label="Check scaled feature",
    )
    scaled_feature_selector
    return (scaled_feature_selector,)


@app.cell
def _(features_scaled_df, px, scaled_feature_selector):
    selected_scaled_feature = scaled_feature_selector.value

    fig_scaled_feature = px.line(
        x=features_scaled_df[selected_scaled_feature].index,
        y=features_scaled_df[selected_scaled_feature],
    )
    fig_scaled_feature.update_layout(
        title=f"Scaled {selected_scaled_feature}", width=800, height=500
    )
    fig_scaled_feature
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
