# DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation (Unofficial Reproduction)

本專案為 **DoubleU-Net** 醫學影像分割論文的非官方復現實作。
專案基於 TensorFlow 2.10.0，針對大腸鏡息肉影像資料集進行精準的病灶分割，並探討在硬體資源受限下的超參數優化策略與模型泛化能力。

---

## 📌 專案簡介 (Overview)

DoubleU-Net 是一種用於醫學影像分割的雙重卷積神經網路架構。本實作完整重現了論文中的核心機制：
1. **Network 1**：採用預訓練的 VGG-19 作為特徵提取器 (Encoder)，初步定位病灶位置。
2. **Network 2**：透過將 Network 1 的輸出結果 (Output 1) 進行二次殘差學習與邊界微調，產出更精細、抗反光干擾的最終遮罩 (Output 2)。
3. **ASPP 模組**：引入空洞空間卷積池化金字塔 (Atrous Spatial Pyramid Pooling)，以捕捉多尺度的上下文特徵。

---

## 📊 資料集 (Dataset)

本專案嚴格遵循論文中的 **實驗 B (Experiment B)** 設定：
* **使用資料集**：CVC-ClinicDB
* **影像數量**：總計 612 張原始內視鏡影像與對應之醫師標註遮罩 (Ground Truth)
* **資料切分**：Training (80%) / Validation (10%) / Testing (10%)

---

## ⚙️ 實作環境與超參數設定 (Implementation Details)

因應本地端硬體資源限制，本實作對原論文的超參數進行了合理化調整，以確保模型能順利訓練並收斂：

| 參數名稱 | 原論文設定 | 本專案實作設定 | 備註說明 |
| :--- | :--- | :--- | :--- |
| **GPU** | N/A (高階多卡) | **NVIDIA RTX 3060 Ti (8GB VRAM)** | 單張消費級顯示卡 |
| **Framework**| Keras / TF | **TensorFlow 2.10.0** | |
| **Image Size**| 384 x 512 | **192 x 256** | 降維以防止 OOM (記憶體溢出) |
| **Batch Size**| 16 | **4** | 因 Batch Size 縮小，權重更新頻率提高 4 倍 |
| **Epochs** | 300 | **100 (Early Stopping)** | 提早停止機制於第 46 回合觸發 |
| **Optimizer**| N/A | **Adam (Learning Rate = 1e-4)** | |
| **Loss** | Dice Loss | **Dice Loss** | |

---

## 🏆 實驗結果 (Results)

在未見過的測試集上，本復現模型展現了與原論文高度一致的優異表現。約 2% 的微小差距主要源於影像降維造成的邊界平滑化，以及小批次更新帶來的收斂軌跡變化。

| Method | DSC (Dice) | mIoU | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- |
| U-Net (Baseline)| 0.8781 | 0.7881 | 0.7865 | 0.9329 |
| DoubleU-Net (Paper)| 0.9239 | 0.8611 | 0.8457 | 0.9592 |
| **Our Reproduction**| **0.9283** | **0.8809** | **0.9165** | **0.9607** |

---

## 📁 專案結構 (Repository Structure)

> **⚠️ 注意事項 (Note):** > 因 GitHub 檔案大小限制，原始資料集 (`dataset/`) 與訓練完畢之權重檔 (`model.h5`) 未包含於本 Repository 中。請自行下載 CVC-ClinicDB 資料集，並依照下方結構放置。

```text
├── dataset/                 # (未上傳) CVC-ClinicDB 資料集請自行放置於此
│   ├── images/              # (未上傳) 原始內視鏡影像 (.png)
│   └── masks/               # (未上傳) 真實解答 Ground Truth (.png)
├── files/                   # 訓練過程日誌與產出目錄
│   ├── data.csv             # 訓練過程數據紀錄 (Loss, Dice 等)
│   └── model.h5             # (未上傳) 訓練完畢之權重，將於執行 train.py 後自動產生
├── model.py                 # DoubleU-Net 網路架構定義
├── metrics.py               # 自定義評估指標與損失函數 (Dice Coef, Dice Loss, mIoU)
├── utils.py                 # 資料處理工具 (影像讀取、資料增強、正規化)
├── train.py                 # 模型訓練腳本 (包含 Early Stopping 設定)
├── evaluate.py              # 測試集量化指標計算 (DSC, mIoU, Recall, Precision)
├── predict.py               # 視覺化預測腳本 (產出影像對比結果)
├── plot_curves.py           # 繪製 Training/Validation Loss 與 Dice 曲線圖
└── README.md                # 專案說明文件
