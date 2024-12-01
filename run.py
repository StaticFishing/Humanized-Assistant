import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import json
import openai


# 加载保存的模型和分词器
MODEL_PATH = "bert_text_classification"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 将模型转移到设备

# 超参数
MAX_LEN = 128
TOP_K = 64  # 输出 Top-k 预测

def load_label_mapping(file_path="idx_label_mapping.json"):
    """
    加载保存的 idx_labels_mapping 文件
    """
    with open(file_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Label mapping has been loaded from {file_path}")
    return {int(key): value for key, value in mapping.items()}  # 将键转换为整数（如果需要）

# 预测函数
def predict(text, model, tokenizer, device, k=64):
    model.eval()
    with torch.no_grad():
        # 对输入文本进行编码
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 模型预测
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # 转为概率分布
        top_k_probs, top_k_indices = torch.topk(probabilities, k=k, dim=1)

        # 将结果转为列表
        top_k_probs = top_k_probs.cpu().numpy()[0]
        top_k_indices = top_k_indices.cpu().numpy()[0]

        return top_k_indices, top_k_probs

label_mapping = load_label_mapping()

properties = {}

with open("motion.json", "r", encoding="utf-8") as f:
    motion = json.load(f)
    for k, v in motion.items():
        properties[int(k)] = v

# 测试示例
label_mapping_inv = {k: v for k, v in label_mapping.items()}  # 反转标签映射

print(label_mapping_inv)

# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = "sk-M4incZE8MEAoNTWw98Df5b0c2c6f4bC19d566bFc37A21914"

# all client options can be configured just like the `OpenAI` instantiation counterpart
openai.base_url = "https://free.v36.cm/v1/"
openai.default_headers = {"x-foo": "true"}

conversation_history = [
    {"role": "system", "content": "你是一个友好的助手，回答问题时严格按照用户要求的格式输出。"},
    # {"role": "system", "content": "请用以下json格式输出：\n- 第一行：语气\n- 第二行：积极，消极，中性选一个词\n你的回答：你的回答在这里"},
    {"role": "system", "content": "请用以下json格式输出：{语气:一种语气 , 标签: 积极,中性,消极其中一个 , 回答: 你的回答}"},
    {"role": "system", "content": "你的回答中不需要加双引号"},
                        ]
print("ChatGPT 已启动！输入 '退出' 以结束对话。\n")

while True:
    text = input("Input：")
    if text == "q":
        break

    conversation_history.append({"role": "user", "content": text})
    # 输出预测结果
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
        )
    except Exception as e:
        print(f"请求失败: {e}")
        continue

    assistant_reply = completion.choices[0].message.content

    print(assistant_reply)

    # reply_split = assistant_reply.split('\n')
    #
    # top_k_indices, top_k_probs = predict(assistant_reply, model, tokenizer, device, k=TOP_K)
    #
    # top_k_indices = zip(top_k_indices, top_k_probs)
    #
    # final_answer = []
    #
    # for k, v in top_k_indices:
    #     if properties[int(k)] in reply_split[1]:
    #         final_answer.append((v,k))
    #
    # final_answer = final_answer[:3]
    #
    # print(reply_split)
    # print("Top-{} Predictions:".format(TOP_K))
    # for i, (prob, idx) in enumerate(final_answer):
    #     try:
    #         print(f"Rank {i + 1}:idx = {idx} Label = {label_mapping_inv[int(idx)]}, Probability = {prob:.4f}")
    #     except KeyError as e:
    #         print(f"KeyError encountered for index {int(idx)}: {e}")
