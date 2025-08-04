import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class RetinaGenerator(Sequence):
    def __init__(self, csv_path, img_dir, mode='binary', augmenter=None, batch_size=32, image_size=(224, 224), shuffle=True, balance_class=False):        
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Disease_Risk'].isin([0, 1])]        
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.mode = mode.lower()
        self.balance_class = balance_class
        
        if self.balance_class:
            self._balance_classes()
        
        if self.mode == 'binary':
            self.label_cols = ['Disease_Risk']
        elif self.mode == 'multi':
            self.label_cols = [col for col in self.df.columns if col not in ['ID']]
        else:
            raise ValueError("Must be binary or multi")
        
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df)) / self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        df = self.df.iloc[batch_indexes]
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.img_dir, str(row["ID"]) + '.png')
            
            #image = load_img(img_path, target_size = self.image_size)
            #image = img_to_array(image)
            
            # FOR TRANSFER LEARNING
            image = load_img(img_path, target_size=self.image_size, color_mode='rgb')
            image = img_to_array(image)
            
            if self.augmenter:
                image = self.augmenter.random_transform(image)
                        
            if self.mode == 'binary':
                label = float(row['Disease_Risk'])
            else:
                label = row[self.label_cols].values.astype(np.float32)
                
            images.append(image)
            labels.append(label)
            
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def _balance_classes(self):
        class0 = self.df[self.df["Disease_Risk"] == 0]
        class1 = self.df[self.df["Disease_Risk"] == 1]
        
        minority, majority = (class0, class1) if len(class0) < len(class1) else (class1, class0)
        upsampled = minority.sample(len(majority), replace=True, random_state=42)
        
        self.df = pd.concat([upsampled, majority]).sample(frac=1, random_state=42)
        self.indexes = np.arange(len(self.df))
                
    def get_balanced_labels(self):
        return self.df['Disease_Risk'].values