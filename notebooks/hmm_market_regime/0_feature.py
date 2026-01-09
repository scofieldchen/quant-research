import marimo

__generated_with = "0.18.4"
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
    import matplotlib.pyplot as plt
    import seaborn as sns

    pio.templates.default = "seaborn"

    data_dir = Path("data")
    return Path, data_dir, mo, pd, plt, px, sns


@app.cell
def _(mo):
    mo.md("""
    ## 研究特征

    ---
    """)
    return


@app.cell
def _(mo):
    file_selector = mo.ui.file(
        filetypes=[".csv"], label="Select training data file", kind="area"
    )
    file_selector
    return (file_selector,)


@app.cell
def _(Path, file_selector, mo):
    mo.stop(not file_selector.value)

    file_path = Path(file_selector.value[0].name)
    mo.md(f"File selected: **{file_path.name}**")
    return (file_path,)


@app.cell
def _(data_dir, file_path, file_selector, mo, pd):
    mo.stop(not file_selector.value)

    # 获取训练集
    features = pd.read_csv(
        data_dir / file_path, index_col="date", parse_dates=True
    )

    # 获取检验集
    asset, *_, wfo_num = file_path.stem.split("_")
    wfo_num = int(wfo_num)

    oos_prices = pd.read_csv(
        data_dir / f"{asset}_oos_prices.csv", index_col="date", parse_dates=True
    )
    oos_prices = oos_prices.query("wfo == @wfo_num")

    # 选择特征
    feature_names = list(features.columns)
    feature_selector = mo.ui.dropdown(
        options=feature_names,
        value=feature_names[0],
        label="Select feature",
    )
    feature_selector
    return feature_selector, features, oos_prices


@app.cell
def _(feature_selector, features, file_selector, mo, oos_prices, pd, px):
    mo.stop(not file_selector.value)

    # 选择特征，合并训练集和检验集
    selected_feature = feature_selector.value
    train_data = features[selected_feature].to_frame().assign(type="train")
    test_data = oos_prices[selected_feature].to_frame().assign(type="test")
    joined = pd.concat([train_data, test_data])

    # 可视化特征
    fig = px.line(
        joined.reset_index(),
        x="date",
        y=selected_feature,
        color="type",
        color_discrete_map={"train": "#1f77b4", "test": "#ff7f0e"},
        labels={"date": "Date", "type": "Data type"},
    )

    fig.update_layout(
        title=f"Train vs Test ({selected_feature})",
        width=800,
        height=500,
    )

    fig
    return joined, selected_feature


@app.cell
def _(file_selector, joined, mo, plt, selected_feature, sns):
    mo.stop(not file_selector.value)

    fig_dist, ax = plt.subplots(figsize=(8, 5))
    ax = sns.histplot(joined, x=selected_feature, hue="type", ax=ax)
    ax.set_title(f"Distribution comparison: {selected_feature}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
