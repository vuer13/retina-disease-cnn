from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import recall_score, f1_score, precision_score
from tensorflow.keras.applications.efficientnet import preprocess_input

import sys
import os

sys.path.append(os.path.abspath('../simplenet'))

from retina import RetinaGenerator
import numpy as np
import pandas as pd
import config

batch_size = 64

def focal_loss(gamma=2.0, alpha=0.5):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

totalVal = len(pd.read_csv(config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels.csv'))
totalTest = len(pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels.csv'))

val_csv = os.path.join(config.DATASET_PATH_VAL, 'RFMiD_Validation_Labels_new.csv')
test_csv = os.path.join(config.DATASET_PATH_TEST, 'RFMiD_Testing_Labels_new.csv')

val_dir = os.path.join(config.DATASET_PATH_VAL, 'Validation')
test_dir = os.path.join(config.DATASET_PATH_TEST, 'Test')

valAug = ImageDataGenerator(preprocessing_function=preprocess_input)
testAug = ImageDataGenerator(preprocessing_function=preprocess_input)

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

model = load_model("../transfer_model/retina_model.h5", custom_objects={"loss_fn": focal_loss()})

val_labels = []
val_preds = []
for i in range(len(valGen)):
    x, y = valGen[i]
    val_labels.extend(y.flatten())
    val_preds.extend(model.predict(x, verbose=0).flatten())

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
best_thresh = thresholds[np.argmax(tpr - fpr)]
best_thresh = best_thresh * 0.8
print(f"Optimal Threshold: {best_thresh:.3f}")

predId = model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > best_thresh).astype("int32")

test_df = pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv')
y_true = test_df["Disease_Risk"].astype("int32").values

print(classification_report(y_true[:len(predId)], predId))
print(confusion_matrix(y_true[:len(predId)], predId))