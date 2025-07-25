from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import recall_score, f1_score, precision_score
from retina import RetinaGenerator
import numpy as np
import pandas as pd
import config
import os

batch_size = 64

def focal_loss(gamma=2.0, alpha=0.45):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

custom_loss = focal_loss(gamma=2.0, alpha=0.45)

totalVal = len(pd.read_csv(config.DATASET_PATH_VAL + '/RFMiD_Validation_Labels.csv'))
totalTest = len(pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels.csv'))

val_csv = os.path.join(config.DATASET_PATH_VAL, 'RFMiD_Validation_Labels_new.csv')
test_csv = os.path.join(config.DATASET_PATH_TEST, 'RFMiD_Testing_Labels_new.csv')

val_dir = os.path.join(config.DATASET_PATH_VAL, 'Validation')
test_dir = os.path.join(config.DATASET_PATH_TEST, 'Test')

valAug = ImageDataGenerator()
testAug = ImageDataGenerator()

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

model = load_model("../final_model/retina_model_50_3.h5", custom_objects={"loss_fn": custom_loss})

val_labels = []
val_preds = []
for i in range(len(valGen)):
    x, y = valGen[i]
    val_labels.extend(y.flatten())
    val_preds.extend(model.predict(x, verbose=0).flatten())

val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

# best_thresh = max(thresholds, key=lambda t: f1_score(val_labels, val_preds > t, pos_label=1))

fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
best_thresh = thresholds[np.argmax(tpr - fpr)]
best_thresh = best_thresh * 0.8

"""
thresholds = np.linspace(0.1, 0.9, 100)
best_thresh = 0.5
best_score = 0
for t in thresholds:
    preds = (val_preds > t).astype("int32")
    pres_1 = precision_score(val_labels, preds, pos_label=1)
    pres_0 = precision_score(val_labels, preds, pos_label=0)
    recall_0 = recall_score(val_labels, preds, pos_label=0)
    recall_1 = recall_score(val_labels, preds, pos_label=1)

    if pres_1 >= 0.9 and recall_0 >= 0.68 and recall_1 >= 0.9 and pres_0 >= 0.6:
        score = f1_score(val_labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
"""

print(f"Optimal Threshold: {best_thresh:.3f}")

predId = model.predict(x=testGen, steps=(totalTest // batch_size) + 1)
predId = (predId > best_thresh).astype("int32")

test_df = pd.read_csv(config.DATASET_PATH_TEST + '/RFMiD_Testing_Labels_new.csv')
y_true = test_df["Disease_Risk"].astype("int32").values

print(classification_report(y_true[:len(predId)], predId))
print(confusion_matrix(y_true[:len(predId)], predId))