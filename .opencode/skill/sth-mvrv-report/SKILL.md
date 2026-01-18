---
name: sth-mvrv-report
description: 生成 sth-mvrv 标准分数动量系统双语分析报告（中文 + 英文）。读取数据和图表，先生成中文报告，再翻译为英文版本，最后保存为 report_zh.md 和 report_en.md。适用于需要更新加密货币市场中期动量分析并发布至中英文平台时。
---

# STH-MVRV 报告生成技能

## 角色定义
作为资深加密货币量化分析师，负责将 sth-mvrv 原始数据与图表转化为具有市场洞察力、专业且引人入胜的双语分析报告，适配社交媒体与深度研究平台。

## 核心流程
请复制并执行以下任务清单，确保每一步都已完成：

```markdown
报告任务进度：
- [ ] 1. 读取数据：加载 `notebooks/sth_mvrv/outputs/summary.md`
- [ ] 2. 视觉分析：读取 `notebooks/sth_mvrv/outputs/` 下的所有 .png 图片并提取关键趋势
- [ ] 3. 撰写中文报告：基于数据与视觉分析结果填充中文模板
- [ ] 4. 翻译为英文：将中文报告内容翻译为英文版本，使用英文模板结构
- [ ] 5. 存储结果：将中文报告保存至 `notebooks/sth_mvrv/report_zh.md`，英文报告保存至 `notebooks/sth_mvrv/report_en.md`
```

## 执行细节

### 1. 输入路径与要求
- **摘要数据**: `notebooks/sth_mvrv/outputs/summary.md`。必须提取最新的 sth-mvrv 数值和标准分数。
- **图表分析**: 遍历 `notebooks/sth_mvrv/outputs/` 目录。
    - 重点观察标准分数是否突破 0 线（多空分界点）。
    - 观察标准分数是否超过 2.0（进入过热区/高位风险）。
    - 观察标准分数是否低于 -2.0（进入恐慌区/可能筑底）。
    - 观察 sth-mvrv 是否在 1.0 附近（市场心理盈亏平衡点）。

### 2. 报告模板 (内容营销版)

#### 中文模板 (Chinese Template)
```markdown
# 🚀 STH-MVRV 动量追踪：{核心结论简述}

## 📊 市场脉搏
**数据日期：** {插入日期}
**当前状态：** {状态标签：如 🟢 强势看涨 / 🟡 高位预警 / 🔴 趋势反转}

### 核心解读：
- **盈亏平衡翻转：** {描述 STH-MVRV 数值及其心理意义，关注 1.0 关口}
- **动量喷发：** {描述标准分数的表现，使用叙事化语言描述趋势强度}
- **风险雷达：** {针对当前数值的风险提示，如过热或背离}

## 🧪 系统表现
> "数据不撒谎，逻辑胜过直觉。"

{描述动量系统的历史战绩，重点突出盈利因子、胜率和平均单笔收益}

---
💡 **分析师笔记：** {插入一段关于未来 1-2 周的预期或操作建议}
```

#### 英文模板 (English Template)
```markdown
# 🚀 STH-MVRV Momentum Tracker: {Core Conclusion}

## 📊 Market Pulse
**Date:** {Date}
**Status:** {Status Tag: e.g., 🟢 Bullish / 🟡 Overheated / 🔴 Reversal}

### Key Insights:
- **Break-even Pivot:** {Describe STH-MVRV value and psychological significance around 1.0}
- **Momentum Surge:** {Describe Z-Score performance with narrative flair}
- **Risk Radar:** {Risk warnings regarding current levels, e.g., overheated or divergence}

## 🧪 Backtest Edge
> "Data doesn't lie; logic beats intuition."

{Performance metrics: Profit Factor, Win Rate, Avg Trade Return}

---
💡 **Analyst's Note:** {Expectations or actionable advice for the next 1-2 weeks}
```

### 3. 约束条件
- **叙事化表达**: 保持专业性的同时，使用动感词汇（CN: “破冰”、“洗盘”; EN: "Breakthrough", "Shakeout", "Overheated"）。
- **严禁编造**: 所有核心数值必须来源于 `summary.md` 或图表观察。
- **翻译要求**: 英文版本必须是中文内容的准确翻译，保持相同的数值和结构，但使用英文叙事风格。
- **保存指令**: 撰写完成后，必须保存两个文件：中文至 `notebooks/sth_mvrv/report_zh.md`，英文至 `notebooks/sth_mvrv/report_en.md`。如果文件已存在，覆盖更新。

## 示例报告 (中文版)

```markdown
# 🚀 STH-MVRV 动量追踪：多头强势，但高位风险积聚

## 📊 市场脉搏
**数据日期：** 2026-01-14
**当前状态：** 🟢 强势看涨（中期） | ⚠️ 短期高热

### 核心解读：
- **盈亏平衡翻转：** STH-MVRV 当前录得 **0.99**。这意味着短期持有者已基本摆脱亏损，市场正处于从"割肉盘"向"获利盘"转换的关键心理关口。
- **动量喷发：** 标准分数（Z-Score）飙升至 **2.8644**。自去年 12 月底"破冰"上穿零线以来，中期动量已彻底激活。多头目前占据绝对统治地位。
- **风险雷达：** 警惕！Z-Score 突破 2.0 通常意味着市场进入"过热区"。历史经验显示，此处极易出现高位横盘或技术性回踩。

## 🧪 系统表现
> "在加密市场，顺势而为是唯一的长久之道。"

我们的 **STH-MVRV 中期动量系统**（>0 做多 / <0 做空）持续证明其捕获趋势的能力：
- **盈利因子 (Profit Factor):** **5.12** —— 这是一个极其强悍的表现，意味着每承担 1 美元风险，可换取 5.12 美元收益。
- **胜率:** 52%
- **平均单笔收益:** +3.46%

---
💡 **分析师笔记：**
中期趋势依然坚挺，但 2.8644 的标准分数暗示"追高"风险正在积聚。建议关注 0 轴支撑，只要标准分数不跌破零线，任何回踩都是健康的洗盘。
```

## 示例报告 (英文版)

```markdown
# 🚀 STH-MVRV Momentum Tracker: Bulls Reclaim Control, But Watch for "Heat" Recoil

## 📊 Market Pulse
**Date:** 2026-01-14
**Status:** 🟢 Bullish (Mid-term) | ⚠️ Short-term Overheated

### Key Insights:
- **Break-even Pivot:** STH-MVRV is currently at **0.99**. This indicates short-term holders are nearly out of the "red," marking a critical psychological shift from capitulation to profit-taking.
- **Momentum Surge:** The Z-Score has surged to **2.86**. Since crossing the zero-line in late December, mid-term momentum has been fully activated. Bulls are firmly in the driver's seat.
- **Risk Radar:** Caution! A Z-Score above 2.0 typically signals an "overheated" zone. Historical data suggests a high probability of consolidation or a technical pullback at these levels.

## 🧪 Backtest Edge
> "In crypto, riding the trend is the only sustainable path."

Our **STH-MVRV Mid-term Momentum System** (>0 Long / <0 Short) continues to prove its edge:
- **Profit Factor:** **5.12** —— A stellar performance, returning $5.12 for every $1 risked.
- **Win Rate:** 52%
- **Avg. Trade Return:** +3.5%

---
💡 **Analyst's Note:** 
The mid-term trend remains robust, but a Z-Score of 2.86 warns that "FOMO" risk is building. Watch the zero-line support; as long as the Z-Score stays positive, any dip is a healthy shakeout.
```
