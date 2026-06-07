import os
import numpy as np
import cv2
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_model
from utils import *
from metrics import *
from tensorflow.keras.callbacks import TerminateOnNaN


def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def iou_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    )
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([288, 384, 3])
    y.set_shape([288, 384, 2])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("files")

    train_path = r"c:\Users\yuan\Desktop\Final_Project\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\dataset\train"
    valid_path = r"c:\Users\yuan\Desktop\Final_Project\DoubleU-Net\2020-CBMS-DoubleU-Net-master\CVC-ClinicDB\dataset\val"

    train_x = sorted(glob(os.path.join(train_path, "images", "*.png")))
    train_y = sorted(glob(os.path.join(train_path, "masks", "*.png")))

    train_x, train_y = shuffling(train_x, train_y)

    valid_x = sorted(glob(os.path.join(valid_path, "images", "*.png")))
    valid_y = sorted(glob(os.path.join(valid_path, "masks", "*.png")))

    print(f"✅ 成功讀取！訓練集圖片: {len(train_x)} 張, 驗證集圖片: {len(valid_x)} 張")

    model_path = "files/model.h5"
    batch_size = 16
    epochs = 300
    lr = 1e-4
    shape = (288, 384, 3)

    model = build_model(shape)
    metrics = [dice_coef, iou_coef, Recall(), Precision()]

    def create_fast_dataset(x, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_fast_dataset(train_x, train_y, batch_size)
    valid_dataset = create_fast_dataset(valid_x, valid_y, batch_size)

    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=20),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True, verbose=1
        ),
    ]

    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle=False,
    )
