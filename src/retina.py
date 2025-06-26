import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class RetinaGenerator(Sequence):
    def __init__(self, csv_path, img_dir, mode='binary', augmenter=None, batch_size=32, image_size=(64, 64), shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.mode = mode.lower()
        
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
    
    def reshuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        df = self.df.iloc[batch_indexes]
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.img_dir, row["ID"] + '.png')
            image = load_img(img_path, target_size = self.image_size)
            image - img_to_array(image)
            
            if self.augmenter:
                image - self.augmenter.random_transform(image)
            
            image = image / 255.0
            label = row[self.label_cols].values.astype(np.float32)
            
            if self.mode == 'binary':
                label = label[0]
                
            images.append(image)
            labels.append(label)
            
        return np.array(images), np.array(labels)
            
        