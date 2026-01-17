# 量化研究项目 AI 协作规范 (AGENTS.md)

## 1. 项目概览
本项目是一个专注于金融市场数据分析、信号挖掘与报告生成的量化研究平台。采用 Python 3.12+ 编写，强调代码的模块化、类型安全与工程稳健性。项目集成了多种数据源（如 Binance, Yahoo Finance, BGeometrics）和分析工具。

## 2. 环境管理与核心指令
本项目使用 `uv` 作为统一的包管理和工作流工具。

### 2.1 环境初始化
- **安装依赖**: `uv sync`
- **更新依赖**: `uv lock --upgrade`
- **运行环境**: 所有指令应在 `uv` 虚拟环境中执行，例如 `uv run python ...`。

### 2.2 构建与校验
在提交任何代码更改前，AI 代理必须执行以下指令以确保代码质量：
- **Ruff 检查**: `uv run ruff check .`
- **自动修复**: `uv run ruff check . --fix`
- **代码格式化**: `uv run ruff format .`

### 2.3 任务运行模式
- **管道任务**: `uv run python -m src.pipelines.<pipeline_name>.task`

## 3. 编码风格与规范

### 3.1 命名约定
- **模块与包**: 使用小写字母和下划线 (`snake_case`)。例如：`src/core/data_loader.py`。
- **类名**: 使用大驼峰命名法 (`PascalCase`)。例如：`class BinanceClient:`。
- **函数与变量**: 使用小写字母和下划线 (`snake_case`)。例如：`def fetch_ohlcv_data(symbol: str):`。
- **常量**: 全大写加下划线 (`UPPER_SNAKE_CASE`)。例如：`MAX_RETRIES = 5`。

### 3.2 类型注解 (Type Hinting)
- **强制性**: 所有新编写的函数和方法必须包含完整的类型注解。
- **现代语法**: 优先使用 Python 3.10+ 的原生类型支持：
  - 使用 `list[int]` 而非 `typing.List[int]`。
  - 使用 `dict[str, Any]` 而非 `typing.Dict`。
  - 使用 `str | None` 代替 `typing.Optional[str]`。
- **复杂类型**: 对于 DataFrame，建议在注释中注明列名结构。

### 3.3 文档字符串 (Docstrings)
- **格式**: 遵循 **Google Style** 格式。
- **要求**: 包含 `Args`, `Returns`, `Raises`(可选,仅在需要时使用)。
- **示例**:
  ```python
  def process_signals(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
      """根据阈值过滤交易信号。

      Args:
          data: 包含信号分数的原始 DataFrame。
          threshold: 过滤的最小阈值。

      Returns:
          仅包含超过阈值信号的 DataFrame。
      """
  ```

### 3.4 导入规范
- **绝对路径**: 始终使用相对于项目根目录的绝对导入。
  - 正确: `from src.core.logger import get_logger`
- **排序**: 按照标准库、第三方库、本地模块的顺序排列（由 Ruff 自动管理）。

## 4. 核心工程模式

### 4.1 结构化日志 (Logging)
- **工具**: 核心日志库为 `loguru`。
- **初始化**: 通过 `src.core.logger.get_logger` 获取。
- **原则**: 严禁在生产代码中使用 `print()`。日志应包含足够的上下文以便调试。

### 4.2 文件与路径处理
- **工具**: 必须使用 `pathlib.Path` 处理所有路径。
- **兼容性**: 避免硬编码路径字符串，确保 Windows/Linux/macOS 兼容。

### 4.3 健壮性与容错 (Retry)
- **网络 I/O**: 任何涉及网络请求的操作必须使用 `tenacity` 进行重试。
- **配置**: 推荐使用指数退避策略，设置合理的 `stop` 和 `wait` 参数。

### 4.4 数据存储与流向
- **格式**: 
  - 原始数据: `.csv` 或 `.json` (data/raw/)
  - 清洗后的数据: `.parquet` (data/cleaned/)，利用其高效的列式存储。
- **一致性**: 确保时间戳列统一命名为 `datetime` 并设为索引。

## 5. 目录职责划分
- `src/core/`: 存放高复用的基础组件（如 `bgeometrics.py` 客户端、`logger.py`）。
- `src/pipelines/`: 存放具体的业务逻辑。每个子目录代表一个完整的数据管道（如 `sth_mvrv/`）。
- `notebooks/`: 交互式研究记录，推荐使用 Marimo notebook。
- `data/`: 数据存储中心，分为 `raw`, `cleaned`, `aggregated` 子目录。

---
*Last updated: 2026-01-17*
