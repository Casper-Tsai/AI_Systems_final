# Burn AI (燒燙傷辨識與醫護指導系統)

本專案是一個基於 AI 的燒燙傷輔助系統，結合了電腦視覺 (Computer Vision) 與大型語言模型 (LLM) 技術。

## 功能說明

1.  **燒燙傷影像辨識**：
    *   使用輕量化深度學習模型 (`model_mobile.ptl`) 辨識燙傷等級。
    *   支援辨識：一級燙傷 (1st Degree)、二級燙傷 (2nd Degree)、三級燙傷 (3rd Degree)。
2.  **智慧醫護建議**：
    *   整合 Google Gemini API。
    *   根據辨識結果、使用者輸入的年齡與受傷原因，提供客製化的急救護理建議。
3.  **多語言支援**：
    *   支援繁體中文 (Traditional Chinese)、英文 (English)、日文 (Japanese) 介面與建議。

## 環境需求

*   Python 3.11.14 (建議使用)
*   ```bash
    conda create -n Burn_AI python=3.11.14
    ```

## 安裝步驟

1.  **複製專案** (或下載檔案)：
    將所有專案檔案下載至本地資料夾。
    ```bash
    git clone https://github.com/your-repo/Burn_AI.git
    cd Burn_AI
    ```

2.  **安裝依賴套件**：
    開啟終端機 (Terminal) 或命令提示字元 (CMD)，切換至專案目錄並執行：
    ```bash
    conda activate Burn_AI
    pip install -r requirements.txt
    ```

3.  **設定 API Key**：
    *   在專案根目錄下建立一個 `.env` 檔案。
    *   將您的 Google Gemini API Key 填入：
        ```env
        GEMINI_API_KEY="Your Gemini API Key"
        ```
    *   若無 API Key，請前往 [Google AI Studio](https://aistudio.google.com/) 申請。

## 使用方式

1.  **啟動程式**：
    執行以下指令開啟應用程式：
    ```bash
    python Burn_AI.py
    ```

2.  **操作介面**：
    *   程式啟動後，點擊終端機顯示的 local URL，如 `http://127.0.0.1:7860`可在本地端開啟介面可使用公開的 public URL，如 `https://d40512dc8fa6a1efbc.gradio.live`可在其他裝置中開啟介面。
    *   選擇語言 (預設為繁體中文)。
    *   點擊「上傳或拍攝照片」提供患部影像。
    *   (選填) 輸入年齡與燙傷原因。
    *   點擊「開始分析」即可查看辨識結果與建議。

## 注意事項

*   **版本相容性**：本專案依賴 `torch` 與 `gradio` 的特定功能，請務必使用 `requirements.txt` 指定的版本範圍，以避免相容性錯誤。
*   **模型檔案**：請確保 `model_mobile.ptl` 位於專案根目錄，否則無法進行影像辨識。
*   **免責聲明**：本系統僅供輔助參考，**絕非專業醫療診斷**。若傷勢嚴重，請立即就醫。
