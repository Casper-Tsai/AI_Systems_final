import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np

# --- 步驟 1: 載入模型 (與之前版本相同) ---

device = torch.device("cpu")
model = efficientnet_b0()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 3)

model_path = "best_model.pt"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"錯誤：找不到模型檔案 '{model_path}'。")
    print("請確認 'best_model.pt' 檔案和此程式在同一個資料夾中。")
    exit()

model.eval()

# --- 步驟 2: 定義影像轉換和類別名稱 (與之前版本相同) ---

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['一級燙傷 (1st Degree)', '二級燙傷 (2nd Degree)', '三級燙傷 (3rd Degree)']

# --- 步驟 3: 新增！處理建議字典 ---

# ‼️ 非常重要的免責聲明
disclaimer = "\n\n**免責聲明：** 本系統建議僅供初步參考，不能取代專業醫療診斷。若情況嚴重或不確定，請立即尋求醫師協助或撥打 119。"

advice_dict = {
    '一級燙傷 (1st Degree)': 
        "### 初步處理建議 (一級燙傷):\n"
        "一級燙傷主要特徵為皮膚發紅、疼痛，無水泡。\n"
        "1.  **降溫**：立即用流動的冷水沖洗傷口至少 10-15 分鐘，以減輕疼痛和降低皮膚溫度。\n"
        "2.  **保濕**：降溫後可塗抹蘆薈膠或溫和的保濕乳液，保持皮膚濕潤。\n"
        "3.  **保護**：避免傷口再次摩擦或受到日曬。"
        + disclaimer,
    
    '二級燙傷 (2nd Degree)':
        "### 初步處理建議 (二級燙傷):\n"
        "二級燙傷特徵為劇烈疼痛、皮膚紅腫並出現水泡。\n"
        "1.  **降溫**：同樣先用流動的冷水沖洗傷口降溫。\n"
        "2.  **保護水泡**：**絕對不要**弄破水泡，以免造成感染。可用無菌紗布或乾淨的布覆蓋傷口。\n"
        "3.  **立即就醫**：二級燙傷有感染風險，強烈建議尋求專業醫師協助，以進行後續處理。"
        + disclaimer,
        
    '三級燙傷 (3rd Degree)':
        "### 緊急處理建議 (三級燙傷):\n"
        "三級燙傷已傷及皮膚深層，可能呈現焦黑或白色，且可能因神經受損而無痛感。\n"
        "1.  **立即撥打 119**：這是嚴重的醫療緊急情況，需要立即送醫。\n"
        "2.  **不要自行處理**：**不要**在傷口上塗抹任何藥膏或冰敷，以免加重傷害。\n"
        "3.  **保持傷口清潔**：可用乾淨的布或紗布覆蓋傷口，等待救護人員到來。\n"
        "4.  **注意休克症狀**：注意患者是否有臉色蒼白、冒冷汗等休克現象。"
        + disclaimer
}

# --- 步驟 4: 修改 predict 函式，讓它同時回傳建議 ---

def predict(image):
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    input_tensor = transform_val(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 準備分類結果
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # 找出信心度最高的類別
    predicted_class = max(confidences, key=confidences.get)
    
    # 從字典中找出對應的建議
    advice = advice_dict[predicted_class]
    
    # 回傳兩個結果：分類信心度和處理建議
    return confidences, advice

# --- 步驟 5: 修改 Gradio 介面，改成上傳模式並顯示建議 ---

with gr.Blocks(theme=gr.themes.Base()) as interface:
    gr.Markdown(
        """
        # 🤖 燙傷分析與初步處置建議系統
        請上傳一張清晰的皮膚患部照片，或使用手機鏡頭拍照。系統將分析可能的燙傷等級並提供初步處理建議。
        """
    )
    with gr.Row():
        # 修改輸入元件，同時允許上傳檔案和使用鏡頭拍照
        image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="上傳或拍攝照片")
        
        with gr.Column():
            # 輸出元件 1: 顯示分類結果
            result_output = gr.Label(num_top_classes=3, label="分析結果")
            # 輸出元件 2: 顯示處理建議
            advice_output = gr.Markdown(label="初步處置建議")
            
    # 設定觸發方式：當 image_input 的內容改變時 (例如上傳了新照片)，就執行 predict
    image_input.change(fn=predict, inputs=image_input, outputs=[result_output, advice_output])

# 啟動 App
interface.launch(share=True)
