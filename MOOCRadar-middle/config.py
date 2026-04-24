import os

# GPT模型配置
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:12345/v1"

# 嵌入模型配置
os.environ["OPENAI_API_KEY"] = "None"
os.environ["EMBEDDING_BASE_URL"] = "http://127.0.0.1:12345/v1"

# 智谱AI模型配置
os.environ["ZHIPUAI_API_KEY"] = "d7916bbd9716431b83c210f1103018d0.afsICgypMiCevFHQ"

# 阿里云千问模型配置
os.environ["DASHSCOPE_API_KEY"] = "sk-3e7cd468c79845cfb981780e2fe5c061"  # 请替换为你的API Key