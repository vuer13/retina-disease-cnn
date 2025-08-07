import sys
import os

sys.path.append(os.path.abspath('../simplenet'))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image, ImageFilter, ImageOps

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve

import config
from retina import RetinaGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

batch_size = 64
img_size = (224, 224)
lr = 1e-5 
epoch = 50

totalTrain = len(pd.read_csv(config.DATASET_PATH_TRAIN + '/RFMiD_Training_Labels.csv'))
totalVal = len(pd.read_csv(config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels.csv'))
totalTest = len(pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels.csv'))

train_csv = os.path.join(config.DATASET_PATH_TRAIN, 'RFMiD_Training_Labels_new.csv')
val_csv = os.path.join(config.DATASET_PATH_VAL, 'RFMiD_Validation_Labels_new.csv')
test_csv = os.path.join(config.DATASET_PATH_TEST, 'RFMiD_Testing_Labels_new.csv')

train_dir = os.path.join(config.DATASET_PATH_TRAIN, 'Training')
val_dir = os.path.join(config.DATASET_PATH_VAL, 'Validation')
test_dir = os.path.join(config.DATASET_PATH_TEST, 'Test')

trainAug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
	rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

valAug = ImageDataGenerator(preprocessing_function=preprocess_input)

testAug = ImageDataGenerator(preprocessing_function=preprocess_input)


trainingGen = RetinaGenerator(
    csv_path = train_csv,
    img_dir = train_dir,
    mode = 'binary',
    augmenter = trainAug,
    batch_size = batch_size,
    image_size = (224, 224),
    shuffle = True,
    balance_class=True
)

valGen = RetinaGenerator(
    csv_path = val_csv,
    img_dir = val_dir,
    mode = 'binary',
    augmenter = valAug,
    batch_size = batch_size,
    image_size = (224, 224),
    shuffle = False,
    balance_class=False
)

testGen = RetinaGenerator(
    csv_path = test_csv,
    img_dir = test_dir,
    mode = 'binary',
    augmenter = testAug,
    batch_size = batch_size,
    image_size = (224, 224),
    shuffle = False
)

train_counts = np.unique(next(iter(trainingGen))[1], return_counts=True)
val_counts = np.unique(next(iter(valGen))[1], return_counts=True)

print(f"Train class distribution: {train_counts}")
print(f"Val class distribution: {val_counts}")

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False

new_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])

class BalancedMetrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        val_labels, val_preds = [], []
        valGen.on_epoch_end()
        for i in range(len(valGen)):
            x, y = valGen[i]
            val_labels.extend(y.flatten())
            val_preds.extend(self.model.predict(x, verbose=0).flatten())
            
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        
        val_preds_class = (val_preds > 0.5).astype(int)
        balanced_acc = balanced_accuracy_score(val_labels, val_preds_class)
        
        logs['val_balanced_acc'] = balanced_acc
        print(f"\nVal Balanced Acc: {balanced_acc:.4f}")

def focal_loss(gamma=2.0, alpha=0.5):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

lr_schedule = CosineDecay(
    initial_learning_rate=lr,
    decay_steps=50*len(trainingGen),
    alpha=0.1
)

opt = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
auc = AUC(name='auc', curve='ROC', num_thresholds=200, multi_label=False)
new_model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy', Recall(), auc, Precision()])
callbacks = [BalancedMetrics(valGen),
             EarlyStopping(monitor='val_balanced_acc', mode='max', patience=20, restore_best_weights=True),
             ModelCheckpoint('../transfer_model/best_model.h5', save_best_only=True, monitor='val_balanced_acc', mode='max', verbose=1),
             CSVLogger('training.log')
]

original_df = pd.read_csv(train_csv)
original_labels = original_df['Disease_Risk'].values

class_weight_dict = {0: 1.0, 1: 4.0}
print(class_weight_dict)

H = new_model.fit(
    x=trainingGen,
    steps_per_epoch=len(trainingGen),
    validation_data=valGen,
    validation_steps=len(valGen),
    epochs=50,
    callbacks=callbacks,
    shuffle=False,
    class_weight=class_weight_dict
)

val_labels = []
val_preds = []
for i in range(len(valGen)):
    x, y = valGen[i]
    val_labels.extend(y.flatten())
    val_preds.extend(new_model.predict(x, verbose=0).flatten())

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
best_thresh = thresholds[np.argmax(tpr - fpr)]
best_thresh = best_thresh
print(f"Optimal Threshold: {best_thresh:.3f}")

predId = new_model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > best_thresh).astype("int32")

test_df = pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv')
y_true = test_df["Disease_Risk"].astype("int32").values

print(classification_report(y_true[:len(predId)], predId))
print(confusion_matrix(y_true[:len(predId)], predId))
new_model.save("../transfer_model/retina_model.h5")
    
N = len(H.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc", linestyle='--')
plt.plot(np.arange(0, N), H.history["val_balanced_acc"], label="val_balanced_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"]) 