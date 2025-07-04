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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import f1_score
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

epoch = 50
lr = 1e-4
batch_size = 32
maxEpoch = epoch

def poly_decay(epoch):
    baseLr = lr
    power = 1.0
    
    alpha = baseLr * (1 - (epoch / float(maxEpoch))) ** power

    return alpha

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
	rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True,
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

# For deduplication - not necessary as their arent actually any duplicates
"""
def get_robust_hash(img_path, hash_size=16):
    try:
        img = Image.open(img_path).convert('L')
        img = Image.open(img_path).resize((256, 256))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Skipping {img_path}: {str(e)}")
        return None

def compare_images(img_path1, img_path2, threshold=0.9):
    try:
        img1 = Image.open(img_path1).resize((256, 256))
        img2 = Image.open(img_path2).resize((256, 256))
        
        # Compare both hash and histogram
        hash_diff = imagehash.phash(img1) - imagehash.phash(img2)
        hist_diff = cv2.compareHist(
            cv2.calcHist([np.array(img1)], [0], None, [256], [0,256]),
            cv2.calcHist([np.array(img2)], [0], None, [256], [0,256]),
            cv2.HISTCMP_CORREL
        )
        return hash_diff < 5 and hist_diff > threshold
    except:
        return False

#DOUBLE CHECK HERE
VAL_CSV_PATH = '../data/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels_new.csv'
VAL_IMG_DIR = '../data/Evaluation_Set/Evaluation_Set/Validation'
TRAIN_IMG_DIR = '../data/Training_Set/Training_Set/Training'

val_df = pd.read_csv(VAL_CSV_PATH)

os.makedirs('./clean_validation/images', exist_ok=True)
clean_csv_path = './clean_validation/validation_clean.csv'

train_hashes = {}
for train_img in os.listdir(TRAIN_IMG_DIR):
    if train_img.endswith('.png'):
        train_path = os.path.join(TRAIN_IMG_DIR, train_img)
        try:
            img = Image.open(train_path).resize((256, 256))
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            train_hashes[train_img] = imagehash.phash(img)
        except Exception as e:
            print(f"Skipping {train_img}: {str(e)}")

valid_samples = []
duplicates_removed = 0
missing_removed = 0

val_df = pd.read_csv(VAL_CSV_PATH)
duplicates = []
valid_samples = []

for _, row in val_df.iterrows():
    img_id = row['ID']
    img_path = os.path.join(VAL_IMG_DIR, f"{img_id}.png")
    
    if not os.path.exists(img_path):
        continue  
        
    try:
        val_img = Image.open(img_path).resize((256, 256))
        val_img = val_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        val_hash = imagehash.phash(val_img)
        
        is_duplicate = any(
            (val_hash - train_hash) < 5  
            for train_hash in train_hashes.values()
        )
        
        if is_duplicate:
            duplicates.append(img_id)
        else:
            valid_samples.append(row)
    except Exception as e:
        print(f"Error processing {img_id}: {str(e)}")

print(f"\nFound {len(duplicates)} duplicates")
clean_df = pd.DataFrame(valid_samples)
clean_df.to_csv('./clean_validation/validation_clean.csv', index=False)

df_cleaned = pd.read_csv('./clean_validation/validation_clean.csv')

df_balanced = df_cleaned.groupby('Disease_Risk').apply(lambda x: x.sample(n=min(len(x), 75), random_state=42)).reset_index(drop=True)
df_balanced = df_balanced.sample(frac = 1, random_state = 42).reset_index(drop=True)
df_balanced.to_csv('./clean_validation/validation_clean.csv', index=False)
""" 
    
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

def focal_loss(gamma=3.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

model = SimpleNet.build(224, 224, 3, classes=1, reg=l2(0.001))
opt = Adam(learning_rate=lr)
model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy', Recall(), AUC(), Precision()])

#callbacks =[LearningRateScheduler(poly_decay), early_stop]
callbacks = [ReduceLROnPlateau(monitor='val_auc', factor = 0.5, patience=3, mode='max'), 
             EarlyStopping(monitor='val_recall', mode='max', patience=5, restore_best_weights=True),
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
class_weight_dict = {0: 3.0, 1: 0.63}
print(class_weight_dict)

#newTotalVal = len(pd.read_csv('./clean_validation/validation_clean.csv'))

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
print(f"Final counts - Labels: {len(val_labels)}, Predictions: {len(val_preds)}")

assert len(val_labels) == len(val_preds), f"Shape mismatch: {len(val_labels)} vs {len(val_preds)}"

thresholds = np.linspace(0.1, 0.9, 25)
best_thresh = max(thresholds, key=lambda t: f1_score(val_labels, val_preds > t, average='weighted'))
print(f"Optimal Threshold: {best_thresh:.3f}")

predId = model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > best_thresh).astype("int32")

test_df = pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv')
y_true = test_df["Disease_Risk"].astype("int32").values
print(np.unique(y_true))

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