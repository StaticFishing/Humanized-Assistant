import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # 添加进度条
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# 数据集示例
data_path = 'data.csv'  # 替换为你的数据路径
df = pd.read_csv(data_path, encoding='UTF-8')  # 根据需要修改编码格式

# 数据准备
texts = df['texts'].tolist()
labels = df['labels'].tolist()
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
df['labels'] = df['labels'].map(label_mapping)
labels = df['labels'].tolist()

import json

# 生成 idx_labels_mapping
idx_labels_mapping = {idx: label for label, idx in label_mapping.items()}

# 保存 idx_labels_mapping 为 JSON 文件
def save_label_mapping(mapping, file_path="decoder_mapping.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    print(f"Label mapping has been saved to {file_path}")

# 调用保存函数
save_label_mapping(idx_labels_mapping)

texts = texts[:1000]
labels = labels[:1000]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# 超参数
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 5e-5
MODEL_NAME = "bert-base-chinese"

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 数据加载器
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 确保使用 GPU（如果可用）
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_mapping))
model.to(device)  # 将模型转移到设备上（GPU 或 CPU）

# 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, desc="Training", leave=True)  # 添加进度条
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)  # 将数据移到 GPU
        attention_mask = batch['attention_mask'].to(device)  # 将数据移到 GPU
        labels = batch['label'].to(device)  # 将标签移到 GPU

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # 更新进度条的后缀信息
        loop.set_postfix(loss=loss.item())
    return total_loss / len(data_loader)

# 验证函数
def eval_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    loop = tqdm(data_loader, desc="Validation", leave=True)  # 添加进度条
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)  # 将数据移到 GPU
            attention_mask = batch['attention_mask'].to(device)  # 将数据移到 GPU
            labels = batch['label'].to(device)  # 将标签移到 GPU

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())  # 转回 CPU
            true_labels.extend(labels.cpu().numpy())  # 转回 CPU
    return accuracy_score(true_labels, predictions)


def eval_model_top_k(model, data_loader, device, k=3):
    model.eval()
    top_k_correct = 0  # Top-k 的正确预测计数
    total_samples = 0  # 样本总数
    loop = tqdm(data_loader, desc="Validation", leave=True)  # 添加进度条

    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            top_k_predictions = torch.topk(logits, k=k, dim=1).indices  # 获取每个样本的 Top-k 预测
            labels = labels.view(-1, 1)  # 将标签 reshape 为列向量

            # 检查正确标签是否在 Top-k 预测中
            top_k_correct += torch.sum(top_k_predictions == labels).item()
            total_samples += labels.size(0)

    # 计算 Top-k 准确率
    top_k_accuracy = top_k_correct / total_samples
    return top_k_accuracy


# 主训练循环
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    top_k_accuracy = eval_model_top_k(model, val_loader, device, k=3)
    print(f"Train Loss: {train_loss:.4f} | Top-3 Validation Accuracy: {top_k_accuracy:.4f}")
# 保存模型
model.save_pretrained("bert_text_classification")
tokenizer.save_pretrained("bert_text_classification")
