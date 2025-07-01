import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Recall, AUC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, RocCurveDisplay
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
lr = 5e-5
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
    
print("checkpoint1")

trainAug = ImageDataGenerator(
	rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.9, 1.1],
    channel_shift_range=20.0,
    fill_mode="reflect"
)

valAug = ImageDataGenerator()

testAug = ImageDataGenerator()



train_csv = os.path.join(config.DATASET_PATH_TRAIN, 'RFMiD_Training_Labels_new.csv')
val_csv = os.path.join(config.DATASET_PATH_VAL, 'RFMiD_Validation_Labels_new.csv')
test_csv = os.path.join(config.DATASET_PATH_TEST, 'RFMiD_Testing_Labels_new.csv')

train_dir = os.path.join(config.DATASET_PATH_TRAIN, 'Training')
val_dir = os.path.join(config.DATASET_PATH_VAL, 'Validation')
test_dir = os.path.join(config.DATASET_PATH_TEST, 'Test')

def get_robust_hash(img_path, hash_size=8):
    try:
        img = Image.open(img_path).resize((256, 256))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Skipping {img_path}: {str(e)}")
        return None


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
            train_hashes[train_img] = imagehash.phash(Image.open(train_path))
        except:
            continue

valid_samples = []
duplicates_removed = 0
missing_removed = 0

print("Processing validation set...")
for idx, row in val_df.iterrows():
    img_id = row['ID']
    img_path = os.path.join(VAL_IMG_DIR, f"{img_id}.png")
    img_file = f"{img_id}.png"
    
    if not os.path.exists(img_path):
        missing_removed += 1
        continue
        
    try:
        val_hash = imagehash.phash(Image.open(img_path))
        is_duplicate = any(val_hash == train_hash for train_hash in train_hashes.values())
        
        if not is_duplicate:
            shutil.copy(img_path, f'./clean_validation/images/{img_id}.png')
            valid_samples.append(row)
        else:
            duplicates_removed += 1
    except:
        continue

clean_df = pd.DataFrame(valid_samples)
clean_df.to_csv(clean_csv_path, index=False)

print("\n=== RESULTS ===")
print(f"Removed duplicates: {duplicates_removed}")
print(f"Removed missing files: {missing_removed}")
print(f"Final clean samples: {len(clean_df)}")

    

    
trainingGen = RetinaGenerator(
    csv_path = train_csv,
    img_dir = train_dir,
    mode = 'binary',
    augmenter = trainAug,
    batch_size = batch_size,
    image_size = (224, 224),
    shuffle = True
)

valGen = RetinaGenerator(
    csv_path = clean_csv_path,
    img_dir = './clean_validation/images',
    mode = 'binary',
    augmenter = valAug,
    batch_size = batch_size,
    image_size = (224, 224),
    shuffle = False
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

print("checkpoint2")

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

X, y = next(iter(valGen))
print("[VAL SANITY CHECK]", np.unique(y, return_counts=True))

model = SimpleNet.build(224, 224, 3, classes=1, reg=l2(0.0001))
opt = Adam(learning_rate=lr)
model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy', Recall(), AUC()])

early_stop = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=10,
    min_delta=0.01,
    baseline=0.6,
    restore_best_weights=True,
    verbose=1
)

#callbacks =[LearningRateScheduler(poly_decay), early_stop]
callbacks = [ReduceLROnPlateau(monitor='val_auc', mode='max', factor = 0.45, patience=3, min_lr=1e-6, verbose=1, min_delta=0.005), 
             early_stop,
             CSVLogger('training.log'),
             ModelCheckpoint('../model/best_model.h5',
                            save_best_only=True,
                            save_weights=False,
                            monitor='val_auc',
                            mode='max',
                            verbose=1)]

train_labels = pd.read_csv(config.DATASET_PATH_TRAIN + '/RFMiD_Training_Labels_new.csv')["Disease_Risk"]
print(train_labels.value_counts(normalize=True))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))
#class_weight_dict = {0: 2.6, 1: 0.75}
print(class_weight_dict)

newTotalVal = len(pd.read_csv('./clean_validation/validation_clean.csv'))

H = model.fit(
    x=trainingGen,
    steps_per_epoch=totalTrain // batch_size,
    validation_data=valGen,
    validation_steps=newTotalVal // batch_size,
    epochs=epoch,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

print("checkpoint3")

predId = model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > 0.3).astype("int32")

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