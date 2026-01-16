import base64
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.logger import get_logger

# 加载环境变量
load_dotenv()
logger = get_logger("sthmvrv")

# 数据目录
INPUT_DIR = "/users/scofield/quant-research/notebooks/sth_mvrv/outputs"
OUTPUT_DIR = "/users/scofield/quant-research/reports/sth_mvrv/drafts"

# 选择支持多模态的模型
MODEL_NAME = "google/gemini-3-flash-preview"

# 系统提示词模板：定义角色和规则
SYSTEM_PROMPT_TEMPLATE = """
你是一位专业的加密货币市场分析师，擅长分析指标如 STH-MVRV Z-score。你的任务是基于提供的 JSON 摘要、CSV 数据和图表，生成简洁的市场洞察报告（Markdown 格式）。重点关注趋势、信号和比特币价格影响。

### 约束条件
1. **风格**：客观、数据驱动、适合社交媒体分享。
2. **长度**：控制在 200-500 字，避免冗长。
3. **内容**：解释 Z-score 值、趋势，并参考图片描述视觉元素（如峰值、谷值）。
4. **格式**：输出纯 Markdown，无额外标记。包括标题、要点和结论。
5. **语言**：使用 {language} 生成报告，专业且易懂。
"""

# 用户提示词模板：传递具体任务参数
USER_PROMPT_TEMPLATE = """
{context}

请基于以上数据和附加图片生成市场洞察，使用 {language}。
"""


# 结构化输出模型
class MarketInsightResponse(BaseModel):
    markdown: str = Field(..., description="生成的 Markdown 格式市场洞察报告。")


class ReportGenerator:
    """
    报告生成器类，用于为不同指标生成市场洞察报告。

    该类读取指定文件夹中的数据文件（JSON、CSV、PNG），使用 LLM 生成 Markdown 报告，
    并提供保存接口。支持多模态输入（图片传递给 LLM）以提高输出质量。

    Attributes:
        data_dir (Path): 数据文件夹路径。
        language (str): 生成内容的语言（例如 "zh" 或 "en"）。
        system_prompt_template (str): 系统提示词模板。
        user_prompt_template (str): 用户提示词模板。
        json_data (list[dict[str, Any]]): 加载的所有 JSON 数据列表。
        csv_data (list[pd.DataFrame]): 加载的所有 CSV 数据列表。
        chart_images (list[str]): Base64 编码的图片列表。
    """

    def __init__(self, data_dir: Path, language: str = "zh"):
        """
        初始化报告生成器。

        Args:
            data_dir: 包含分析数据的文件夹路径（例如 notebooks/sth_mvrv/outputs）。
            language: 生成内容的语言（例如 "zh" 为中文, "en" 为英文）。默认为 "zh"。

        Raises:
            FileNotFoundError: 如果文件夹不存在。
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"数据文件夹不存在: {data_dir}")
        if language not in ("zh", "en"):
            raise ValueError("语言必须为中文(zh)或者英文(en)")

        self.data_dir = data_dir
        self.language = language
        self.system_prompt_template = SYSTEM_PROMPT_TEMPLATE
        self.user_prompt_template = USER_PROMPT_TEMPLATE
        self.json_data: list[dict[str, Any]] = []
        self.csv_data: list[pd.DataFrame] = []
        self.chart_images: list[str] = []

        self._load_data()

    def _load_data(self) -> None:
        """读取文件夹中的所有数据文件（JSON、CSV、PNG）。"""
        try:
            # 读取所有 JSON 文件
            json_files = list(self.data_dir.glob("*.json"))
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    self.json_data.append(json.load(f))
                logger.info(f"加载 JSON 数据: {json_file}")

            # 读取所有 CSV 文件
            csv_files = list(self.data_dir.glob("*.csv"))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                self.csv_data.append(df)
                logger.info(f"加载 CSV 数据: {csv_file}")

            # 读取所有 PNG 文件（编码为 Base64）
            png_files = list(self.data_dir.glob("*.png"))
            for png_file in png_files:
                with open(png_file, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    self.chart_images.append(f"data:image/png;base64,{encoded}")
                logger.info(f"加载图片: {png_file}")

        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            raise

    def _format_data_for_prompt(self) -> str:
        """
        格式化 JSON 和 CSV 数据为字符串，用于插入用户提示词。

        分别处理每个 JSON 和每个 CSV，不合并，以避免字段冲突。

        Returns:
            格式化后的文本字符串。
        """
        context_parts = []

        # 格式化每个 JSON 数据
        for i, data in enumerate(self.json_data):
            context_parts.append(f"### JSON 数据 {i+1}\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        # 格式化每个 CSV 数据（提取最后 10 行）
        for i, df in enumerate(self.csv_data):
            csv_str = df.tail(10).to_string(index=False)
            context_parts.append(f"### CSV 数据 {i+1}（最后 10 行）\n{csv_str}")

        return "\n\n".join(context_parts)

    def generate_report(self) -> dict[str, Any]:
        """
        生成市场洞察报告。

        使用 LLM 基于加载的数据生成 Markdown 报告，并返回图表路径列表。

        Returns:
            包含 'markdown' (str) 和 'charts' (list[Path]) 的字典。

        Raises:
            Exception: 如果 LLM 调用失败。
        """
        # 获取格式化后的用户提示词文本
        context = self._format_data_for_prompt()

        # 初始化 LLM
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.5,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            timeout=60,
            max_retries=3,
        )

        # 构建消息（多模态：文本 + 图片）
        messages = [
            ("system", self.system_prompt_template.format(language=self.language)),
            HumanMessage(
                content=[
                    {"type": "text", "text": self.user_prompt_template.format(context=context, language=self.language)},
                ]
                + [
                    {"type": "image_url", "image_url": {"url": img}}
                    for img in self.chart_images
                ]
            ),
        ]

        # 创建链并调用
        prompt = ChatPromptTemplate.from_messages(messages)
        structured_llm = llm.with_structured_output(MarketInsightResponse)
        chain = prompt | structured_llm

        try:
            response = chain.invoke({})
            markdown = response.markdown
            # 图表路径：返回数据文件夹中的 PNG 文件路径
            charts = list(self.data_dir.glob("*.png"))
            logger.info("报告生成成功。")
            return {"markdown": markdown, "charts": charts}
        except Exception as e:
            logger.error(f"生成报告时出错: {e}")
            raise

    def save_to_directory(self, output_dir: Path, report: dict[str, Any]) -> None:
        """
        将报告和图表保存到指定目录。

        Args:
            output_dir: 输出目录路径。
            report: generate_report() 返回的字典，包含 'markdown' 和 'charts'。

        Raises:
            Exception: 如果保存失败。
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存 Markdown
            markdown_file = output_dir / "market_insight.md"
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(report["markdown"])
            logger.info(f"保存 Markdown: {markdown_file}")

            # 复制图表
            for chart_path in report["charts"]:
                dest = output_dir / chart_path.name
                dest.write_bytes(chart_path.read_bytes())
                logger.info(f"复制图表: {dest}")

        except Exception as e:
            logger.error(f"保存报告时出错: {e}")
            raise


if __name__ == "__main__":
    generator = ReportGenerator(Path(INPUT_DIR), language="zh")
    report = generator.generate_report()
    generator.save_to_directory(Path(OUTPUT_DIR), report)
