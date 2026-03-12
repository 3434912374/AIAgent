import os
import requests
from dotenv import load_dotenv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

load_dotenv()
WEATHER_API_KEY=os.getenv("WEATHER_API_KEY")

class WeatherInput(BaseModel):
    # OpenWeatherMap 对拼音或英文的城市名支持更好
    location: str = Field(description="需要查询天气的城市名称，建议使用拼音或英文，例如：Beijing, Tokyo, New York")

class WeatherTool(BaseTool):
    name:str = "get_current_weather"
    description:str = "当你想查询某个城市的当前真实天气时，使用此工具。"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        # 从环境变量获取 API Key，如果没有设置，为了方便你测试，这里做了 fallback 降级直接使用你的 key
      
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        # 构造请求参数
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric", # 直接请求摄氏度 (metric)
            "lang": "zh_cn"    # 让 API 直接返回中文的天气描述
        }
        
        try:
            # 发送真实的网络请求
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status() # 如果返回 4xx 或 5xx 错误，抛出异常
            data = response.json()
            
            # 解析核心数据
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            
            return f"【OpenWeatherMap真实数据】{location}当前天气：{weather_desc}，实际温度 {temp}°C，体感温度 {feels_like}°C，相对湿度 {humidity}%。"
            
        except requests.exceptions.RequestException as e:
            return f"获取 {location} 天气失败，网络请求错误: {e}"
        except KeyError:
            return f"获取 {location} 天气失败，无法解析返回的数据。请确保城市名称正确（建议使用拼音，如 Shanghai）。"