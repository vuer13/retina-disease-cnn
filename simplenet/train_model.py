import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Recall, AUC, Precision
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

from retina import RetinaGenerator
from simplenet import SimpleNet
import config
import tqdm as tqdm

from PIL import Image, ImageFilter, ImageOps
import shutil
import imagehash
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

epoch = 40
lr = 1e-4
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
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
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

def focal_loss(gamma=2.0, alpha=0.5):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

steps_per_epoch = len(trainingGen)

lr_schedule = CosineDecayRestarts(
    initial_lr_rate = lr,
    first_decay_steps = 10 * steps_per_epoch,
    t_mul = 1.0,
    m_mul = 0.9
)

model = SimpleNet.build(224, 224, 3, classes=1, reg=l2(0.001))
opt = Adam(learning_rate=lr_schedule, global_clipnorm = 1.0)
model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy', Recall(), AUC(), Precision()])

#callbacks =[LearningRateScheduler(poly_decay), early_stop]
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=3, min_lr = 1e-7), 
             EarlyStopping(monitor='val_auc', mode='max', patience=10, baseline=0.7, restore_best_weights=True),
             CSVLogger('training.log'),
             ModelCheckpoint('../model/best_model.h5', save_best_only=True, save_weights=False, monitor='val_auc', mode='max', verbose=1)]

original_df = pd.read_csv(train_csv)
original_labels = original_df['Disease_Risk'].values
#train_labels = trainingGen.get_balanced_labels()

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(original_labels),
    y=original_labels
)
#class_weight_dict = dict(zip(np.unique(original_labels), class_weights))
#class_weight_dict = {0: 4.0, 1: 0.5}
#print(class_weight_dict)

H = model.fit(
    x=trainingGen,
    steps_per_epoch=len(trainingGen),
    validation_data=valGen,
    validation_steps=len(valGen),
    epochs=epoch,
    callbacks=callbacks,
    shuffle=False
)

val_labels = []
val_preds = []
for i in range(len(valGen)):
    x, y = valGen[i]
    val_labels.extend(y.flatten())
    val_preds.extend(model.predict(x, verbose=0).flatten())

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

thresholds = np.linspace(0.3, 0.7, 50) 
best_thresh = max(thresholds, key=lambda t: f1_score(val_labels, val_preds > t, pos_label=1))
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
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])