import google.generativeai as genai
import os

# 記得設定你的 API KEY
os.environ["GOOGLE_API_KEY"] = "你的_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("可用模型列表：")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")