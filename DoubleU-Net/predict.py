import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from train import tf_dataset


def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def evaluate_normal(model, x_data, y_data):
    THRESHOLD = 0.5
    total = []
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        _, h, w, _ = x.shape

        y_pred1 = parse(model.predict(x)[0][..., -2])
        y_pred2 = parse(model.predict(x)[0][..., -1])

        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x[0] * 255.0,
            line,
            mask_to_3d(y) * 255.0,
            line,
            mask_to_3d(y_pred1) * 255.0,
            line,
            mask_to_3d(y_pred2) * 255.0,
        ]
        mask = np.concatenate(all_images, axis=1)

        cv2.imwrite(f"results/{i}.png", mask)


smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred):
    smooth_iou = 1e-7
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    )
    return (intersection + smooth_iou) / (union + smooth_iou)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results/")

    batch_size = 8

    test_path = r"C:\Users\yuan\Desktop\Final_Project\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\dataset\test"

    if not os.path.exists(test_path):
        print(f"❌ 錯誤：路徑不存在: {test_path}")
    else:
        print(f"✅ 路徑存在，正在檢查 images 資料夾...")
        image_dir = os.path.join(test_path, "images")
        if os.path.exists(image_dir):
            files = os.listdir(image_dir)
            print(f"   images 資料夾內找到 {len(files)} 個檔案")
            print(f"   檔案範例: {files[:5]}")
        else:
            print(f"❌ 錯誤：找不到 {image_dir} 資料夾！")

    test_x = sorted(glob(os.path.join(test_path, "images", "*.png")))
    if len(test_x) == 0:
        test_x = sorted(glob(os.path.join(test_path, "images", "*.jpg")))

    test_y = sorted(glob(os.path.join(test_path, "masks", "*.png")))
    if len(test_y) == 0:
        test_y = sorted(glob(os.path.join(test_path, "masks", "*.jpg")))

    print(f"最後確認 - 找到影像: {len(test_x)}, 遮罩: {len(test_y)}")
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = len(test_x) // batch_size
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope(
        {"dice_loss": dice_loss, "dice_coef": dice_coef, "iou_coef": iou_coef}
    ):
        model = load_model_weight("files/model.h5")

    metrics = [
        dice_coef,
        iou_coef,
        tf.keras.metrics.Recall(name="Recall"),
        tf.keras.metrics.Precision(name="Precision"),
    ]

    model.compile(optimizer="adam", loss=dice_loss, metrics=metrics)

    print("正在計算各項指標...")
    results = model.evaluate(test_dataset, steps=test_steps, verbose=1)

    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test DSC: {results[1]:.4f}")
    print(f"Test mIoU (Custom): {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test Precision: {results[4]:.4f}")

    evaluate_normal(model, test_x, test_y)
