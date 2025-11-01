import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# 1. 建立和訓練時一模一樣的模型架構
model = efficientnet_b0()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 3)

# 2. 載入訓練好的權重
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

# 3. 將模型轉換為 TorchScript 格式
scripted_model = torch.jit.script(model)

# 4. 優化並儲存為 PyTorch Lite 格式 (.ptl)
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("model_mobile.ptl")

print("模型已成功轉換為 model_mobile.ptl！")
