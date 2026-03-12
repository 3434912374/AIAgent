from langchain.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """当你需要获取关于时事、最新事件或你不知道的实时信息时，请使用此网络搜索引擎。"""
    try:
        with DDGS() as ddgs:
            # 限制返回 3 条结果，既节省 Token 又保证信息量
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "没有搜到相关结果。"
            
            # 将结果拼成字符串
            res_text = "\n".join([f"标题: {r['title']}\n摘要: {r['body']}\n链接: {r['href']}" for r in results])
            return res_text
    except Exception as e:
        return f"搜索过程中出错: {e}"

def get_search_tool():
    return web_search