import os
import matplotlib.pyplot as plt
import pandas as pd

# 讀取 CSV 紀錄檔
csv_path = "files/data.csv"
if not os.path.exists(csv_path):
    print("找不到 data.csv")
    exit()

df = pd.read_csv(csv_path)

plt.figure(figsize=(14, 6))

# ---- 第一張圖：Loss 曲線 (也就是您現在貼的這張) ----
plt.subplot(1, 2, 1)
plt.plot(df["epoch"] + 1, df["loss"], "b-", label="Training Loss")
plt.plot(df["epoch"] + 1, df["val_loss"], "r-", label="Validation Loss")
plt.title("Training and Validation Loss", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.legend()

# ---- 第二張圖：Dice 準確率曲線 ----
plt.subplot(1, 2, 2)
# 請確認您的 csv 欄位名稱是 dice_coef 還是其他名稱，若不同請修改這裡
plt.plot(df["epoch"] + 1, df["dice_coef"], "b-", label="Training Dice")
plt.plot(df["epoch"] + 1, df["val_dice_coef"], "r-", label="Validation Dice")
plt.title("Training and Validation Dice Coefficient", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Dice Score", fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("files/training_curves.png", dpi=300)
plt.show()