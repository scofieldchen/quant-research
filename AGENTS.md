# 量化研究项目 AI 协作规范 (AGENTS.md)

专注于金融数据分析与内容生成。

## 环境管理

- 使用 `uv` 作为统一的包管理和工作流工具。
- 运行python脚本：`uv run python ...`
- 运行管道任务：`uv run python -m src.pipelines.<pipeline_name>.task`

## 项目结构

- `src/`: 数据管道核心逻辑。负责数据的获取，清洗和存储。
- `notebooks/`: 数据探索与信号生成。**必须使用 Marimo notebook**。
- `data/`: 存储数据。
  - `data/raw`：存储原始数据，保留原始数据格式如csv或者json。
  - `data/cleaned`：存储清洗数据，使用`parquet`格式。
  - `data/aggregated`：存储最终数据，例如notebook计算的指标，信号或者模型。

## Python 编码规范

### 1. 代码风格与版本
- **现代语法**: 使用 Python 3.10+ 标准。优先使用内置类型注解（如 `list[str]`, `dict[str, int]`, `str | None`），弃用 `typing.List`, `typing.Union` 等旧写法。
- **质量标准**: 代码需具备生产级健壮性，逻辑清晰，模块化。

### 2. 类型提示
- **强制执行**: 所有函数参数、返回值、类属性**必须**包含类型注解。
- **目的**: 确保代码可通过静态分析，明确数据契约。

### 3. 文档与注释
- **Google Style Docstrings**: 所有模块、类、函数必须包含文档字符串。
  - 必须包含 `Args`, `Returns`，如有异常抛出需包含 `Raises`。
- **行内注释**: 仅在逻辑复杂时添加。解释为什么这么做 (Why)，而不是在做什么 (How)。
- **语言**：文档和注释均使用中文。

### 4. 异常处理
- **精准捕获**: 严禁使用空的 `try...except` 或捕获宽泛的 `Exception`（除非在顶层入口）。
- **IO 容错**: 在涉及网络请求（如数据下载）或批量处理循环中，必须包含异常处理逻辑，确保单个失败不会导致整个进程崩溃。如需重试机制，使用 `tenacity` 实现。
