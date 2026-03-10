import os
from src.utils.decorators import agent_tool_logger


class FileTool:
    """文件操作工具，支持 Agent 进行文件读写"""

    def __init__(self, root_dir: str = "logs/"):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    @agent_tool_logger
    def save_analysis_result(self, filename: str, content: str) -> str:
        "安全写入分析结果到本地文件"
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"分析结果已安全写入文件：{file_path}"
