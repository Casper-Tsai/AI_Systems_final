import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import collections

# --- Part 1 & 2: 載入並整合所有資料集 ---

image_label_list = []

# 從 './archive' 資料夾載入資料
data_dir = './archive'
valid_exts = ['.png', '.jpg', '.jpeg']

if os.path.exists(data_dir):
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            base_name = fname.replace('.txt', '')
            matched_img = None
            for ext in valid_exts:
                candidate = os.path.join(data_dir, base_name + ext)
                if os.path.exists(candidate):
                    matched_img = candidate
                    break
            
            if matched_img is None:
                continue

            txt_path = os.path.join(data_dir, fname)
            with open(txt_path, 'r') as f:
                class_ids = [int(line.split()[0]) for line in f.readlines()]
            
            if class_ids:
                max_class = max(class_ids)
                image_label_list.append((matched_img, max_class))
    print(f"從 '{data_dir}' 載入了 {len(image_label_list)} 張圖片。")
else:
    print(f"警告: 找不到 '{data_dir}' 資料夾，已跳過。")

# 從 './Skin-Burns--2' 資料夾載入並整合 Roboflow 資料
rf_base_dir = "./Skin-Burns--2"
splits = ["train", "valid", "test"]

if os.path.exists(rf_base_dir):
    initial_count = len(image_label_list)
    for split in splits:
        split_dir = os.path.join(rf_base_dir, split)
        # 注意：此處依賴一個特定的 _classes.csv 檔案格式
        csv_path = os.path.join(split_dir, "_classes.csv")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        for _, row in df.iterrows():
            fname = row['filename'].strip()
            img_path = os.path.join(split_dir, fname)

            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if not os.path.exists(img_path):
                continue
            
            # 根據您的原始碼邏輯，從 '0', '1', '2' 欄位中提取標籤
            try:
                label = int(row[['0', '1', '2']].astype(int).idxmax())
                image_label_list.append((img_path, label))
            except Exception as e:
                print(f"處理檔案 {fname} 時出錯: {e}")

    print(f"從 '{rf_base_dir}' 載入了 {len(image_label_list) - initial_count} 張圖片。")
    print(f"總圖片數: {len(image_label_list)}")
else:
    print(f"警告: 找不到 '{rf_base_dir}' 資料夾，已跳過。")


# --- Part 3: 視覺化驗證 ---

# 1. 將圖片路徑按標籤 (0, 1, 2) 進行分組
grouped_images = collections.defaultdict(list)
for img_path, label in image_label_list:
    grouped_images[label].append(img_path)

# 按照標籤排序，確保顯示順序是 0, 1, 2
sorted_labels = sorted(grouped_images.keys())

# 2. 設定我們要顯示的圖片數量
num_classes = len(sorted_labels)
samples_per_class = 5  # 每個類別顯示 5 張範例圖片

# 3. 建立一個畫布來顯示圖片
fig, axs = plt.subplots(num_classes, samples_per_class, figsize=(15, 8))
fig.suptitle("資料集各類別圖片範例 (眼見為憑)", fontsize=20)

# 4. 迭代每個類別並顯示圖片
for i, label in enumerate(sorted_labels):
    # 從該類別的圖片列表中隨機挑選
    if len(grouped_images[label]) >= samples_per_class:
        sample_images = random.sample(grouped_images[label], samples_per_class)
    else:
        # 如果該類別圖片不足，則全部顯示
        sample_images = grouped_images[label]

    for j, img_path in enumerate(sample_images):
        ax = axs[i, j]
        if j == 0:
            ax.set_title(f"標籤 (Label): {label}", fontsize=14)
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off') # 隱藏座標軸
        except Exception as e:
            print(f"無法讀取圖片 {img_path}: {e}")
            ax.text(0.5, 0.5, '無法讀取', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局以容納主標題
plt.show()

