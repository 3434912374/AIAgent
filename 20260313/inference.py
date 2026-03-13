import torch
from models.transformer import TextTransformer

def load_inference_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载保存的“大脑”数据（包含权重和词表）
    checkpoint = torch.load("model_weights.pth", map_location=device)
    vocab = checkpoint['vocab']
    
    # 2. 重建模型结构 (必须与训练时参数完全一致)
    # 这里 vocab_size 使用保存的词表大小
    model = TextTransformer(
        vocab_size=len(vocab), 
        embed_dim=64, 
        nhead=4, 
        num_layers=2, 
        num_classes=2
    ).to(device)
    
    # 3. 灌输“记忆”
    model.load_state_dict(checkpoint['model'])
    model.eval() # 开启预测模式，关闭 Dropout
    
    return model, vocab, device

def predict_sentiment(text, model, vocab, device, max_len=20):
    # 4. 对输入文本进行和训练时一样的预处理
    tokens = text.lower().split()
    # 将单词转为数字，不在词表里的转为 1 (<UNK>)
    seq = [vocab.get(t, 1) for t in tokens]
    # 填充或截断
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
        
    input_tensor = torch.tensor([seq]).to(device) # 增加 batch 维度
    
    # 5. 推理预测
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    return "😊 正面评价 (Positive)" if prediction == 1 else "😡 负面评价 (Negative)"

if __name__ == "__main__":
    # 启动引擎
    print("正在加载模型...")
    model, vocab, device = load_inference_engine()
    
    # 测试环节
    test_sentences = [
        "我很喜欢这部电影！",  
        "我不喜欢洗衣服",
        "这个人太坏了",
        "我不上学了"
    ]
    
    print("\n--- 情感分析测试结果 ---")
    for s in test_sentences:
        res = predict_sentiment(s, model, vocab, device)
        print(f"输入: {s}  =>  预测结果: {res}")