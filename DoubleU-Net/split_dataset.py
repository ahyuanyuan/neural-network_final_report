import os
import shutil
from sklearn.model_selection import train_test_split

image_dir = r"c:\Users\yuan\Desktop\期末論文\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\Original"
mask_dir = r"c:\Users\yuan\Desktop\期末論文\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\Ground Truth"
base_out_dir = r"c:\Users\yuan\Desktop\期末論文\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\dataset"

if os.path.exists(base_out_dir):
    print("發現舊的 dataset 資料夾，正在強制徹底刪除中...")
    shutil.rmtree(base_out_dir)
    print("舊資料夾已清除完畢！")


def create_dir_structure(base_path):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, "masks"), exist_ok=True)


create_dir_structure(base_out_dir)


image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".tif"))]
)
mask_files = sorted(
    [f for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg", ".tif"))]
)

if len(image_files) != len(mask_files):
    print("警告：影像與遮罩的數量不一致！請檢查資料夾。")

# 80/10/10
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=0.5, random_state=42
)


def copy_files(img_list, mask_list, split_name):
    print(f"👉 正在處理 {split_name} 資料集 ({len(img_list)} 張圖片)...")
    for img_name, mask_name in zip(img_list, mask_list):
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(base_out_dir, split_name, "images", img_name)
        shutil.copy(src_img, dst_img)

        src_mask = os.path.join(mask_dir, mask_name)
        dst_mask = os.path.join(base_out_dir, split_name, "masks", mask_name)
        shutil.copy(src_mask, dst_mask)


copy_files(train_imgs, train_masks, "train")
copy_files(val_imgs, val_masks, "val")
copy_files(test_imgs, test_masks, "test")

print("✅ 資料集切割與整理完成！請去檢查 dataset 資料夾！")
