import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from hmmlearn import hmm

    data_dir = Path("data/")
    return StandardScaler, data_dir, hmm, mo, pd


@app.cell
def _(mo):
    mo.md("""
    ## 模型比较

    ---
    """)
    return


@app.cell
def _(StandardScaler, hmm, pd):
    def train_hmm(
        features: pd.DataFrame,
        states: int = 3,
        random_state: int = 123,
        verbose: bool = False,
    ) -> tuple[hmm.GaussianHMM, StandardScaler]:
        """训练隐马尔可夫模型"""
        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 创建模型对象
        model = hmm.GaussianHMM(
            n_components=states,
            covariance_type="full",  # 特征相关程度高，应使用完全协方差矩阵，否则模型可能不收敛
            n_iter=2000,
            tol=1e-5,
            random_state=random_state,
            verbose=verbose,
        )

        # 训练模型
        model.fit(features_scaled)

        return model, scaler
    return (train_hmm,)


@app.cell
def _(mo):
    form = (
        mo.md("""
        **设置参数**
    
        {asset_selector}

        {wfo_nums_selector}

        {states_slider}
        """)
        .batch(
            asset_selector=mo.ui.dropdown(
                options=["BTCUSDT"], value="BTCUSDT", label="选择交易对"
            ),
            wfo_nums_selector=mo.ui.multiselect(
                options=list(range(1, 10)),
                value=[1, 3, 5, 7, 9],
                label="选择训练集",
            ),
            states_slider=mo.ui.range_slider(
                start=3,
                stop=8,
                step=1,
                value=[3, 6],
                show_value=True,
                label="选择状态数范围",
            ),
        )
        .form()
    )

    form
    return (form,)


@app.cell
def _(data_dir, form, mo, pd, train_hmm):
    mo.stop(not form.value)


    # 获取参数
    asset = form.value["asset_selector"]
    wfo_nums = sorted(form.value["wfo_nums_selector"])
    states_range = form.value["states_slider"]
    states = list(range(states_range[0], states_range[1] + 1))

    # 记录BIC
    bic = {}

    # 对每一轮训练集，尝试不同的状态数量，记录每个状态对应的AIC和BIC

    total_iterations = len(wfo_nums) * len(states)

    with mo.status.progress_bar(
        total=total_iterations, title="Train HMM models"
    ) as bar:
        for wfo_num in wfo_nums:
            file_path = data_dir / f"{asset}_train_{wfo_num}.csv"
            features = pd.read_csv(file_path, index_col="date", parse_dates=True)
            bic[wfo_num] = {}

            for state in states:
                bar.update(increment=1, subtitle=f"WFO={wfo_num} States={state}")
                try:
                    model, scaler = train_hmm(features, states=state)
                    features_scaled = scaler.transform(features)
                    model_bic = model.bic(features_scaled)
                except Exception as e:
                    print(e)
                else:
                    bic[wfo_num].update({state: model_bic})
    return bic, states


@app.cell
def _(bic, form, mo, pd, states):
    mo.stop(not form.value)

    # 创建表格显示BIC
    bic_df = pd.DataFrame(bic).T
    bic_df.index.name = "wfo_num"
    bic_df.columns = [f"state({col})" for col in bic_df.columns]

    bic_table = mo.ui.table(
        bic_df,
        show_column_summaries=False,
        show_data_types=False,
        selection=None,
        format_mapping={col: "{:.1f}" for col in bic_df.columns},
    )

    # 创建表格显示BIC的变化
    bic_chg_df = bic_df.pct_change(axis=1).dropna(axis=1)
    bic_chg_df.columns = [
        f"state({states[i - 1]}->{states[i]})" for i in range(1, len(states))
    ]


    def style_cell(_rowId, _columnName, value):
        # 计算柱状条的宽度（基于绝对值的百分比）
        # 假设最大变化为80%，用于标准化柱状条长度
        max_change = 0.8
        bar_width = min(abs(value) / max_change * 100, 100)

        # 根据正负值选择颜色
        if value < 0:
            bar_color = (
                "rgba(76, 175, 80, 0.3)"  # 绿色（负值表示BIC下降，模型改进）
            )
        else:
            bar_color = (
                "rgba(244, 67, 54, 0.3)"  # 红色（正值表示BIC上升，模型变差）
            )

        style = {
            "background": f"linear-gradient(to right, {bar_color} {bar_width}%, transparent {bar_width}%)",
            "backgroundSize": "100% 70%",  # 柱状条高度为单元格的70%
            "backgroundPosition": "center",  # 垂直居中
            "backgroundRepeat": "no-repeat",
        }

        return style


    bic_chg_table = mo.ui.table(
        bic_chg_df,
        show_column_summaries=False,
        show_data_types=False,
        selection=None,
        format_mapping={col: "{:.1%}" for col in bic_chg_df.columns},
        style_cell=style_cell,
    )

    mo.vstack([mo.md("### BIC"), mo.md("---"), bic_table, bic_chg_table])
    return


if __name__ == "__main__":
    app.run()
