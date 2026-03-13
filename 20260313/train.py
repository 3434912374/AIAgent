import torch
import torch.nn as nn
from core.optimizers import CustomAdam
from models.transformer import TextTransformer
from utils.data_loader import get_loaders

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取数据、词表大小和处理器
    loader, vocab_size, processor = get_loaders()
    
    model = TextTransformer(vocab_size, embed_dim=64, nhead=4, num_layers=2, num_classes=2).to(device)
    
    # --- 这里就是你问的“从哪里改”的核心位置 ---
    # 实验建议：改变 betas 观察 Loss 波动
    optimizer = CustomAdam(
        model.parameters(), 
        lr=0.001, 
        betas=(0.9, 0.999), # <--- 尝试修改这里，比如 (0.5, 0.8)
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    print("开始训练真实文本分类器...")
    for epoch in range(20): # 增加循环次数，因为真实数据需要更多学习
        total_loss = 0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save({'model': model.state_dict(), 'vocab': processor.vocab}, "model_weights.pth")
    print("模型与词表已保存。")

if __name__ == "__main__":
    train()