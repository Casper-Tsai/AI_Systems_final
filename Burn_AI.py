import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# --- 1. 環境與設定 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("警告: 未偵測到 GEMINI_API_KEY，請檢查 .env 檔案。")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# 設定裝置
device = torch.device("cpu")

# --- 2. 載入模型 ---
# 使用 TorchScript 載入優化後的移動端模型
MODEL_PATH = "model_mobile.ptl"
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print(f"成功載入模型: {MODEL_PATH}")
except Exception as e:
    print(f"錯誤: 無法載入模型 {MODEL_PATH}。請確認檔案是否存在。錯誤訊息: {e}")
    model = None

# 影像前處理 (需與訓練時一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['1st Degree (一級燙傷)', '2nd Degree (二級燙傷)', '3rd Degree (三級燙傷)']

# --- 3. 多語言介面文字 ---
LANG_TEXT = {
    "繁體中文": {
        "title": "🔥 燒燙傷等級辨識與醫護指導系統",
        "header_desc": "請上傳燙傷照片或使用鏡頭拍攝，系統將分析燙傷等級並提供醫護建議。",
        "input_label": "上傳或拍攝照片",
        "age_label": "年齡",
        "age_placeholder": "選填",
        "cause_label": "燙傷原因",
        "cause_placeholder": "選填，例如：熱水、化學物質、火",
        "analyze_btn": "開始分析",
        "result_label": "辨識結果",
        "advice_label": "醫護建議",
        "loading": "分析中，請稍候...",
        "error_no_image": "請先提供照片！",
        "error_model": "模型未載入，無法分析。",
        "disclaimer": """
### ⚠️ 免責聲明
本系統僅供輔助參考，**絕非專業醫療診斷**。
辨識結果可能存在誤差，若傷勢嚴重、範圍廣大或位於臉部、關節等重要部位，**請立即就醫或撥打緊急救護電話**。
使用本系統即代表您同意自行承擔相關風險。
"""
    },
    "English": {
        "title": "🔥 Burn Injury Classification & Medical Advice System",
        "header_desc": "Upload a photo or use camera. The system will classify the burn degree and provide advice.",
        "input_label": "Upload or Capture Image",
        "age_label": "Age",
        "age_placeholder": "Optional",
        "cause_label": "Cause of Burn",
        "cause_placeholder": "Optional, e.g., Hot Water, Chemical, Fire",
        "analyze_btn": "Analyze",
        "result_label": "Classification Result",
        "advice_label": "Medical Advice",
        "loading": "Analyzing, please wait...",
        "error_no_image": "Please provide an image first!",
        "error_model": "Model not loaded.",
        "disclaimer": """
### ⚠️ Disclaimer
This system is for reference only and **is NOT a professional medical diagnosis**.
Results may be inaccurate. If the injury is severe, extensive, or on sensitive areas (face, joints), **seek immediate medical attention**.
By using this system, you agree to assume all related risks.
"""
    },
    "日本語": {
        "title": "🔥 熱傷深度判定・応急処置アドバイスシステム",
        "header_desc": "写真をアップロードまたは撮影してください。熱傷深度を判定し、アドバイスを提供します。",
        "input_label": "写真のアップロードまたは撮影",
        "age_label": "年齢",
        "age_placeholder": "任意",
        "cause_label": "受傷原因",
        "cause_placeholder": "任意、例：熱湯、化学物質、火",
        "analyze_btn": "分析開始",
        "result_label": "判定結果",
        "advice_label": "医療アドバイス",
        "loading": "分析中、お待ちください...",
        "error_no_image": "写真をアップロードしてください！",
        "error_model": "モデルが読み込まれていません。",
        "disclaimer": """
### ⚠️ 免責事項
本システムは参考用であり、**専門的な医療診断ではありません**。
判定結果には誤差が生じる可能性があります。重症の場合や、顔・関節などの重要部位の場合は、**直ちに医師の診察を受けてください**。
本システムの利用により生じたリスクは、利用者が負うものとします。
"""
    }
}

# --- 4. 核心邏輯 ---

def get_gemini_advice(burn_degree, age, cause, language):
    """使用 Gemini API 生成建議"""
    if not GEMINI_API_KEY:
        return "Error: API Key not found."

    model_gemini = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    lang_prompt = {
        "繁體中文": "請用繁體中文回答。",
        "English": "Please answer in English.",
        "日本語": "日本語で答えてください。"
    }
    
    prompt = f"""
    You are a medical assistant expert in burn care.
    Patient Info:
    - Burn Degree: {burn_degree}
    - Age: {age if age else "Unknown"}
    - Cause: {cause if cause else "Unknown"}
    
    Task:
    1. Explain what this burn degree means.
    2. Provide immediate first aid steps.
    3. Advise on whether to see a doctor immediately.
    4. Give specific advice based on age and cause if provided.
    
    {lang_prompt.get(language, "Please answer in Traditional Chinese.")}
    Keep the response concise, structured (use Markdown), and empathetic.
    Start with a warning/disclaimer.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

def process_analysis(image, age, cause, language):
    """處理分析流程"""
    txt = LANG_TEXT[language]
    
    if image is None:
        return None, txt["error_no_image"]
    
    if model is None:
        return None, txt["error_model"]

    # 1. 影像辨識
    try:
        pil_image = Image.fromarray(image).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 取得最高信心度的類別
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_class = class_names[top_catid[0].item()]
        confidence = top_prob[0].item()
        
        result_str = f"{predicted_class} ({confidence:.1%})"
        
    except Exception as e:
        return f"Error: {e}", "Classification Failed"

    # 2. LLM 建議
    advice = get_gemini_advice(predicted_class, age, cause, language)
    
    return result_str, advice

def update_ui_language(language):
    """更新介面語言文字"""
    t = LANG_TEXT[language]
    return (
        gr.update(value=t["title"]),
        gr.update(value=t["header_desc"]),
        gr.update(label=t["input_label"]),
        gr.update(label=t["age_label"], placeholder=t["age_placeholder"]),
        gr.update(label=t["cause_label"], placeholder=t["cause_placeholder"]),
        gr.update(value=t["analyze_btn"]),
        gr.update(label=t["result_label"]),
        gr.update(label=t["advice_label"]),
        gr.update(value=t["disclaimer"])
    )

def clear_inputs():
    return None, "", "", None, ""

# --- 5. 建構 Gradio 介面 ---
with gr.Blocks() as demo:
    
    # 狀態變數
    current_lang = gr.State("繁體中文")
    
    # 標題區 (使用 HTML 標籤加粗加大)
    title_md = gr.Markdown(f"<h1><b>{LANG_TEXT['繁體中文']['title']}</b></h1>")
    desc_md = gr.Markdown(LANG_TEXT["繁體中文"]["header_desc"])
    
    with gr.Row():
        lang_dropdown = gr.Dropdown(
            choices=["繁體中文", "English", "日本語"],
            value="繁體中文",
            label="Language / 語言 / 言語",
            interactive=True
        )
    
    with gr.Row():
        # 左側輸入區
        with gr.Column(scale=1):
            img_input = gr.Image(sources=["upload", "webcam"], type="numpy", label=LANG_TEXT["繁體中文"]["input_label"])
            age_input = gr.Textbox(label=LANG_TEXT["繁體中文"]["age_label"], placeholder=LANG_TEXT["繁體中文"]["age_placeholder"])
            cause_input = gr.Textbox(label=LANG_TEXT["繁體中文"]["cause_label"], placeholder=LANG_TEXT["繁體中文"]["cause_placeholder"])
            analyze_btn = gr.Button(LANG_TEXT["繁體中文"]["analyze_btn"], variant="primary")
            clear_btn = gr.Button("Clear / 清除")
            
        # 右側輸出區
        with gr.Column(scale=1):
            result_output = gr.Label(label=LANG_TEXT["繁體中文"]["result_label"])
            advice_output = gr.Markdown(label=LANG_TEXT["繁體中文"]["advice_label"])
            
    # 免責聲明
    disclaimer_md = gr.Markdown(LANG_TEXT["繁體中文"]["disclaimer"])

    # --- 事件綁定 ---
    
    # 語言切換
    def update_ui_language_wrapper(language):
        updates = update_ui_language(language)
        t = LANG_TEXT[language]
        return (
            f"<h1><b>{t['title']}</b></h1>",
            t["header_desc"],
            gr.update(label=t["input_label"]),
            gr.update(label=t["age_label"], placeholder=t["age_placeholder"]),
            gr.update(label=t["cause_label"], placeholder=t["cause_placeholder"]),
            gr.update(value=t["analyze_btn"]),
            gr.update(label=t["result_label"]),
            gr.update(label=t["advice_label"]),
            t["disclaimer"]
        )

    lang_dropdown.change(
        fn=update_ui_language_wrapper,
        inputs=[lang_dropdown],
        outputs=[title_md, desc_md, img_input, age_input, cause_input, analyze_btn, result_output, advice_output, disclaimer_md]
    )
    
    # 分析按鈕
    analyze_btn.click(
        fn=process_analysis,
        inputs=[img_input, age_input, cause_input, lang_dropdown],
        outputs=[result_output, advice_output]
    )
    
    # 清除按鈕
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[img_input, age_input, cause_input, result_output, advice_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
