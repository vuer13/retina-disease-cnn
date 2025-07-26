# Retina Disease CNN Classifier

This project uses a Convolutional Neural Network (CNN) to classify retinal images as healthy (0) or diseased (1) using the RFMiD dataset. It includes focal loss for handling class imbalance and custom threshold tuning to reduce false negatives while maintaining strong performance.

---

## Features
- CNN-based binary classifier trained on the RFMiD retinal image dataset
- Focal loss implementation to handle class imbalance
- Custom RetinaGenerator to load image batches from CSV and image directories

---

Model 1: 
Weights - Class 0: 1.5, Class 1: 1.0
Optimal Threshold - 0.373

                precision    recall  f1-score   support

           0       0.66      0.71      0.68       134
           1       0.92      0.90      0.91       506

    accuracy                            0.86       640 
    macro avg       0.79      0.81      0.80       640
    weighted avg    0.87      0.86      0.86       640

10.0% FN, 29.1% FP


Model 2:
Weights - Class 0: 5.0, Class 1: 1.0
Optimal Threshold - 0.431

                precision    recall  f1-score   support

           0       0.65      0.70      0.67       134
           1       0.92      0.90      0.91       506

    accuracy                            0.86       640
    macro avg       0.78      0.80      0.79       640
    weighted avg    0.86      0.86      0.86       640

 10.0% FN, 29.8% FP

This model prioritizes reducing false negatives over false positives, as it is clinically safer to mistakenly flag a healthy retina for review than to miss a diseased retina and leave it undetected.

---

## Challenges:
- Class 0 (non-diseased) had significantly fewer samples than class 1 (diseased), so I applied resampling to balance the training set. As a result, the model may have reduced performance when classifying healthy retinas due to limited representative data

---

## Future Plans
- Develop a frontend interface to allow users to upload and classify their own retinal images
- Implement transfer learning to improve model performance and training efficiency
- Reduce false negatives and improve recall for class 0 (healthy); this will require collecting more healthy retinal images for training
- Extend the model to classify specific retinal diseases
- Find a better dataset to classify diabetic retinopathy specifically