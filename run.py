import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import json
import openai
import time

# 加载保存的模型和分词器
MODEL_PATH = "bert_text_classification"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 将模型转移到设备

# 超参数
MAX_LEN = 128
TOP_K = 64  # 输出 Top-k 预测

def load_label_mapping(file_path="bert_text_classification/idx_label_mapping.json"):
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

def screening(text,emotion):
    top_k_indices, top_k_probs = predict(text, model, tokenizer, device, k=TOP_K)
    top_k_indices = zip(top_k_indices, top_k_probs)
    final_answer = []
    for k, v in top_k_indices:
        if properties[int(k)] in emotion:
            final_answer.append((v,k))

        final_answer = final_answer[:3]
    return final_answer

# 初始化上下文历史和系统消息
conversation_history = [
    {"role": "system", "content": "你是一个友好的助手，回答问题时严格按照用户要求的格式输出。"},
    {"role": "system", "content": "请用以下json格式输出：{语气: 你认为需要对我表达的语气（英语）, 标签: 积极,中性,消极其中一个 , 回答: 你的回答（中文）}"},
    {"role": "system", "content": "要记得加双引号"},
    {"role": "system", "content": "用户输入的格式为{语气: 一种语气, 内容: 用户的输入}"}
]
MAX_HISTORY = 6  # 最大对话轮数限制
def ask_GPT(question):
    global conversation_history  # 声明使用全局变量
    # 更新对话历史
    conversation_history.append({"role": "user", "content": question})
    conversation_history = conversation_history[-MAX_HISTORY:]  # 限制对话长度

    # 调用 OpenAI API 获取模型回复
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
            )
            assistant_reply = completion.choices[0].message.content
            break  # 如果请求成功，退出重试循环
        except Exception as e:
            print(f"请求失败: {e}")
            if attempt < retry_attempts - 1:
                print("正在重试...")
                time.sleep(1)  # 等待1秒后重试
            else:
                print("多次尝试后请求失败，跳过本次输入。")
                assistant_reply = None
    return assistant_reply

def analyze_reply(assistant_reply):
    if assistant_reply is None:
        return None
    result_moji = []
    result_text = []
    try:
        output = json.loads(assistant_reply.replace("\n", "").replace("“", "\"").replace("”", "\""))
        if isinstance(output, dict) and all(key in output for key in ["语气", "标签", "回答"]):
            emoji = screening(output['回答'],output['标签'])
            for e in emoji:
                result_moji.append(label_mapping_inv[e[1]])
            result_text.append(output['回答'])
        else:
            print("模型返回的 JSON 不符合预期格式：", assistant_reply)
    except json.JSONDecodeError:
        print("解析 JSON 失败，模型返回的内容可能不符合格式：", assistant_reply)
    return result_text, result_moji

label_mapping = load_label_mapping()

properties = {} # 筛掉和语义相反的emoji
with open("bert_text_classification/motion.json", "r", encoding="utf-8") as f:
    motion = json.load(f)
    for k, v in motion.items():
        properties[int(k)] = v

label_mapping_inv = {k: v for k, v in label_mapping.items()}  # 反转标签映射

print(label_mapping_inv)

# api-key
openai.api_key = "sk-M4incZE8MEAoNTWw98Df5b0c2c6f4bC19d566bFc37A21914"

# openai的配置
openai.base_url = "https://free.v36.cm/v1/"
openai.default_headers = {"x-foo": "true"}

print("ChatGPT 已启动！输入 'q' 以结束对话。\n")

def process_vocal(vocal):
    ##传入一段语音，返回识别的情感、提取的文本
    ##{语气: 一种语气, 内容: 用户的输入}
    pass


def process_result(dict):
    ##dict是词典格式 格式{"label":***,"text":***}
    ##返回生成的语音
    ##语音保存在./flask/uploads/audio.wav里
    pass

def gpt_api(text):
    assistant_reply = ask_GPT(text)
    # print(assistant_reply)
    # {"语气": "", "标签": "", "回答": ""}回答的格式是这样的字典
    result_text, result_moji = analyze_reply(assistant_reply)
    result = ''
    result += ''.join(map(str, result_moji))
    result += ''.join(map(str, result_text))
    print(result)
    return result



#-------------------------------------主程序--------------------------------------
user_ask = process_vocal()
gpt_answer = gpt_api(user_ask)
process_result(gpt_answer)

