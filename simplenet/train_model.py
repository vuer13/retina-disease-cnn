import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Recall, AUC, Precision
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import config
import tqdm as tqdm
from PIL import Image, ImageFilter, ImageOps
import shutil
import imagehash
import os
from imutils import paths

from retina import RetinaGenerator
from simplenet import SimpleNet

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

epoch = 50
lr = 8e-5
batch_size = 64
maxEpoch = epoch

"""
def poly_decay(epoch):
    baseLr = lr
    power = 1.0
    
    alpha = baseLr * (1 - (epoch / float(maxEpoch))) ** power

    return alpha
"""

totalTrain = len(pd.read_csv(config.DATASET_PATH_TRAIN + '/RFMiD_Training_Labels.csv'))
totalVal = len(pd.read_csv(config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels.csv'))
totalTest = len(pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels.csv'))

datasets = {config.DATASET_PATH_TRAIN + '/RFMiD_Training_Labels.csv' : config.DATASET_PATH_TRAIN + '/RFMiD_Training_Labels_new.csv', 
            config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels.csv' : config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels_new.csv',
            config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels.csv' : config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv'}

for data, new in datasets.items():
    df = pd.read_csv(data)
    df = df.sample(frac = 1, random_state = 42).reset_index(drop=True)
    df.to_csv(new)
    
trainAug = ImageDataGenerator(
	rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1, 
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode="constant"
)

valAug = ImageDataGenerator()

testAug = ImageDataGenerator()

train_csv = os.path.join(config.DATASET_PATH_TRAIN, 'RFMiD_Training_Labels_new.csv')
val_csv = os.path.join(config.DATASET_PATH_VAL, 'RFMiD_Validation_Labels_new.csv')
test_csv = os.path.join(config.DATASET_PATH_TEST, 'RFMiD_Testing_Labels_new.csv')

train_dir = os.path.join(config.DATASET_PATH_TRAIN, 'Training')
val_dir = os.path.join(config.DATASET_PATH_VAL, 'Validation')
test_dir = os.path.join(config.DATASET_PATH_TEST, 'Test')
    
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

def focal_loss(gamma=2.0, alpha=0.45):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn
"""
def lr_schedule(epoch):
    min_lr = lr * 0.01
    if epoch < 15:
        return lr * (epoch + 1) / 15
    elif 15 <= epoch < 30:
        return lr
    else:
        return max(lr - (lr - min_lr) * ((epoch - 30) / 20), min_lr)
    
"""
lr_schedule = CosineDecay(
    initial_learning_rate=lr,
    decay_steps=epoch*len(trainingGen),
    alpha=0.1
)


model = SimpleNet.build(224, 224, 3, classes=1, reg=l2(0.001))
opt = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
# opt = Adam(learning_rate=lr, global_clipnorm = 0.5)
auc = AUC(name='auc', curve='ROC', num_thresholds=200, multi_label=False)
model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy', Recall(), auc, Precision()])

#callbacks =[LearningRateScheduler(poly_decay), early_stop]
callbacks = [# ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=5, min_lr = 1e-6, verbose=1), 
             BalancedMetrics(valGen),
             # LearningRateScheduler(lr_schedule),
             EarlyStopping(monitor='val_balanced_acc', mode='max', patience=20, restore_best_weights=True),
             ModelCheckpoint('../model/best_model.h5', save_best_only=True, save_weights=False, monitor='val_balanced_acc', mode='max', verbose=1),
             CSVLogger('training.log')
]

original_df = pd.read_csv(train_csv)
original_labels = original_df['Disease_Risk'].values
#train_labels = trainingGen.get_balanced_labels()

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(original_labels),
    y=original_labels
)
#class_weight_dict = dict(zip(np.unique(original_labels), class_weights))
class_weight_dict = {0: 6.0, 1: 1.0}
print(class_weight_dict)

H = model.fit(
    x=trainingGen,
    steps_per_epoch=len(trainingGen),
    validation_data=valGen,
    validation_steps=len(valGen),
    epochs=epoch,
    callbacks=callbacks,
    shuffle=False,
    class_weight=class_weight_dict
)

val_labels = []
val_preds = []
for i in range(len(valGen)):
    x, y = valGen[i]
    val_labels.extend(y.flatten())
    val_preds.extend(model.predict(x, verbose=0).flatten())

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

# thresholds = np.linspace(0.35, 0.9, 100) 
# best_thresh = max(thresholds, key=lambda t: f1_score(val_labels, val_preds > t, pos_label=1))
fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
best_thresh = thresholds[np.argmax(tpr - fpr)]
best_thresh = best_thresh * 0.85
print(f"Optimal Threshold: {best_thresh:.3f}")

predId = model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > best_thresh).astype("int32")

test_df = pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv')
y_true = test_df["Disease_Risk"].astype("int32").values

print(classification_report(y_true[:len(predId)], predId))
print(confusion_matrix(y_true[:len(predId)], predId))
model.save("../model/retina_model.h5")

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