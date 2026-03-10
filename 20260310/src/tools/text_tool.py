from src.utils.decorators import agent_tool_logger


class TextTool:
    """文件清洗与分析工具"""

    @agent_tool_logger
    def extract_unique_keywords(self, text: str) -> list:
        """从文本中提取唯一的关键词(去重)"""
        words = text.lower().replace(",", "").replace(".", "").split()
        # 使用set去重后转换为列表
        unique_words = list(set(words))
        # 返回排序后的唯一关键词列表
        return sorted(unique_words)

    @agent_tool_logger
    def word_frequency_stats(self, text: str) -> dict:
        """词频统计（使用字典推导式）"""
        #将整个文本转换为小写
        words = text.lower().split()
        return {word: words.count(word) for word in set(words) if len(word) >= 2}
