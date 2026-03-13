import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

class TextProcessor:
    def __init__(self, texts, max_vocab=1000, max_len=20):
        self.max_len = max_len
        # 1. 构建词频统计
        all_words = " ".join(texts).lower().split()
        counts = Counter(all_words)
        # 2. 选取最高频的词，预留 0 给 padding, 1 给 unknown
        self.vocab = {word: i+2 for i, (word, _) in enumerate(counts.most_common(max_vocab-2))}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        # 将文本转为数字序列，并进行截断或填充
        tokens = text.lower().split()
        seq = [self.vocab.get(t, 1) for t in tokens]
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq)

class RealTextDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = [processor.encode(t) for t in texts]
        self.labels = [torch.tensor(l) for l in labels]

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def get_loaders(batch_size=4):
    # 模拟一些真实的语料（正面 vs 负面）
    raw_texts = [
        "this movie is great", "i love this film", "excellent work", "so good",
        "this is a bad movie", "i hate this film", "terrible acting", "so boring"
    ]
    raw_labels = [1, 1, 1, 1, 0, 0, 0, 0] # 1: 正面, 0: 负面
    
    processor = TextProcessor(raw_texts)
    dataset = RealTextDataset(raw_texts, raw_labels, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, len(processor.vocab), processor