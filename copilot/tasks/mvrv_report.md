## 问题：如何生成sth-mvrv分析报告？

我们建立了数据管道，获取sth-mvrv数据，然后建立marimo notebook，分析sth-mvrv数据，接下来要生成分析报告。

该报告将发布到推特，substack等平台，目标是通过发布有价值的市场洞察来获取粉丝和订阅。

产品定位：内容应该形成差异化竞争，首先专注于加密货币，其次要传递有价值的信号，例如展示经过验证的逻辑。

如何设计技术架构？

工作流设计：

第一步：先用marimo notebook分析数据，制定分析逻辑，生成交易信号，然后将相关数据保存到本地。

```
notebooks/
  sth_mvrv/
    analysis.py
    output/
      summary.json --> 关键数据和信号
      data.csv --> 附加数据
      chart_1.png --> 图片1
      chart_2.png --> 图片2
```

`summary.json`：尽可能添加更多有效的信息。

```json
{
  "date": "2024-05-20",
  "current_btcusd_price": 67000,
  "sth_mvrv": 1.2,
  "sth_mvrv_zscore": 0.8,
  "regime": "Momentum Up"
}
```

第二步：用llm生成内容。

在项目根目录下创建新目录`report`，专门用于生成分析报告。

```
project_root/
  data/           # 存储数据
  src/            # 数据管道
  notebooks/      # marimo notebook，探索数据，生成信号
  report/         # 生成分析报告
    sth_mvrv/       # 为每个指标/模型创建单独的目录
      generate_report.py
      draft/
        twitter.md
        newsletter.md
    sentiment/
      generate_report.py
    ...
```

根据平台类型来设计不同的提示词，生成不同风格的内容
1. 社媒平台（推特，币安广场）
  - 风格：紧迫感，表情符号，关键数据前置。
  - 重点："Z-Score 刚刚突破 0.5，历史上这通常意味着..."
2. 博客平台（substack, publish0x, paragraph）
  - 风格: 专业、教育性、逻辑推演、宏观背景结合。
  - 重点: 解释指标原理，展示指标图表，分析信号和风险。

### 实现第一步：生成中间产物

修改marimo notebook，将分析结果保存到本地，包含关键汇总信息，附加数据和图片。

先输出最核心的数据，先尝试生成报告，根据报告质量来修改内容。

### 实现第二步：生成内容

解决问题：使用llm生成关于sth-mvrv标准分数的市场洞察，并发布到twitter和币安广场等社交媒体。

输入数据存储在`notebooks/sth_mvrv/outputs`目录，包含：

```
summary.json  # 核心指标和信号
sth_mvrv_zscore.csv  # 附加数据,最后几行
indicator_chart.png  # 核心指标图表
```

这些数据应该作为背景资料传递给llm模型。

使用langchain实现交互，统一使用openrouter api, api信息存储在项目根目录下的`.env`文件。

使用langchain与llm交互时需遵循以下原则：
- 使用系统提示词定义角色和规则
- 使用用户提示词定义具体任务（传递参数）
- 使用结构化输出

接下来要生成更长篇的文章，发布到substack等平台。如何修改代码？我认为只需要使用不同的系统和用户提示词，并复用`reportGenerator`类。
