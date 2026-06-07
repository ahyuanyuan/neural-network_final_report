# Neural Network-Final-Report
# DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation (CVC-ClinicDB)
## 實驗結果對照 (CVC-ClinicDB)

以下為本模型在 10% 獨立測試集上的評估結果，與原論文 Table III 的數據對照：

| 評估指標 | 論文官方數據 (Table III) | 本專案複現數據 |
| :--- | :--- | :--- |
| **DSC** | 0.9239 | 0.9358 |
| **mIoU** | 0.8611 | 0.87xx (請填入你最終跑出的數字) |
| **Recall** | 0.8457 | 0.9307 |
| **Precision** | 0.9592 | 0.9363 |

> 視覺化對照圖 (Input vs Ground Truth vs Predict) 可於 `results/` 資料夾中查看。

## 資料夾與檔案結構

* `model.py`: DoubleU-Net 模型架構定義。
* `metrics.py`: 存放自定義評估指標 (`dice_coef`, `iou_coef`, `dice_loss`)。
* `utils.py`: 影像前處理與工具函數。
* `split_dataset.py`: 用於將原始資料集切割為 80% 訓練集、10% 驗證集、10% 測試集。
* `train.py`: 模型訓練腳本。
* `predict.py`: 載入訓練好的權重進行測試，並輸出指標與預測遮罩圖。
* `files/`: 存放訓練過程的 Log (`data.csv`) 與最佳權重檔 (`model.h5`)。
* `results/`: 存放測試集輸出的視覺化結果圖。

## 資料集準備指南

1. [請前往 CVC-ClinicDB 官方來源或 Kaggle 下載原始資料集。 ](https://www.kaggle.com/datasets/balraj98/cvcclinicdb?resource=download)
