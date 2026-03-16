#文件路径
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor,ConsoleSpanExporter

def setup_telemetry():
    """初始化分布式链路追踪，在控制台打印每个组件的绝对耗时"""
    provider=TracerProvider()#创建一个追踪提供者
    processor=SimpleSpanProcessor(ConsoleSpanExporter())#创建一个简单的跨度处理器，使用控制台导出器将跨度信息输出到控制台
    provider.add_span_processor(processor)#将处理器添加到追踪提供者中
    trace.set_tracer_provider(provider)#设置全局追踪提供者，使得后续创建的追踪器都使用这个提供者
    return trace.get_tracer("rag_system_tracer")#创建一个追踪器实例，命名为"rag_system_tracer"，用于在系统中创建跨度并记录追踪信息