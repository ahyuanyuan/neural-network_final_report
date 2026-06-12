
import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

# 1. 設定測試集路徑 (請確認這裡的路徑與您實際情況相符)
test_x_path = "new_data/test/image/"
test_y_path = "new_data/test/mask/"
model_path = "files/model.h5"

print(f"正在載入模型: {model_path} ...")
# 加上 compile=False，避免載入自定義指標時發生錯誤
model = load_model(model_path, compile=False)

# 抓取測試集所有圖片的檔名
test_images = sorted(os.listdir(test_x_path))
num_images = len(test_images)

# 初始化各項指標的加總值
total_dice = 0.0
total_iou = 0.0
total_recall = 0.0
total_precision = 0.0

print(f"開始評估 {num_images} 張測試圖片...")

for img_name in tqdm(test_images):
    # 讀取並正規化原圖
    x = cv2.imread(os.path.join(test_x_path, img_name), cv2.IMREAD_COLOR)
    x = x / 255.0
    x = np.expand_dims(x, axis=0) # 加上 batch 維度: (1, 192, 256, 3)

    # 讀取並正規化真實解答 (Ground Truth)
    y_true = cv2.imread(os.path.join(test_y_path, img_name), cv2.IMREAD_GRAYSCALE)
    y_true = y_true / 255.0
    y_true = y_true > 0.5  # 轉為布林值 (0 或 1)

    # 進行預測
    predictions = model.predict(x, verbose=0)
    y_pred = predictions[0] # 此時形狀為 (192, 256, 2)

    # 🚨 破解 9% 分數的真相：分離 Output 1 與 Output 2
    if len(y_pred.shape) == 3 and y_pred.shape[-1] == 2:
        # 我們不要 argmax 了！我們直接「抽出」第二層 (index 1) 的最終預測圖
        y_pred = y_pred[..., 1] 
    else:
        y_pred = np.squeeze(y_pred) # 防呆機制

    y_pred = y_pred > 0.5  # 轉為布林值確保計算正確

    # 計算交集與聯集
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    tp = np.sum(intersection)              # True Positives (正確抓到的息肉)
    fp = np.sum(y_pred) - tp               # False Positives (誤判的背景)
    fn = np.sum(y_true) - tp               # False Negatives (漏抓的息肉)

    # 為了避免分母為 0，加上一個極小的數值 smooth
    smooth = 1e-5

    # 計算四大指標
    dice = (2.0 * tp + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    iou = (tp + smooth) / (np.sum(union) + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    # 加總
    total_dice += dice
    total_iou += iou
    total_recall += recall
    total_precision += precision

# 計算平均值
avg_dice = total_dice / num_images
avg_iou = total_iou / num_images
avg_recall = total_recall / num_images
avg_precision = total_precision / num_images

# 漂亮地印出成績單
print("\n" + "="*50)
print("🏆 實驗 B (CVC-ClinicDB) 測試集最終評估成績單")
print("="*50)
print(f"1. DSC (Dice 係數) : {avg_dice:.4f}  ({avg_dice*100:.2f}%)")
print(f"2. mIoU (平均交併比): {avg_iou:.4f}  ({avg_iou*100:.2f}%)")
print(f"3. Recall (召回率)  : {avg_recall:.4f}  ({avg_recall*100:.2f}%)")
print(f"4. Precision(精確率): {avg_precision:.4f}  ({avg_precision*100:.2f}%)")
print("="*50)
print("💡 備註：此為模型在完全未見過的 Test Set 上的客觀表現！")