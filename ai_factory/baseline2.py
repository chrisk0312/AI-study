import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_lists
import joblib

# Define paths using pathlib for better OS compatibility
data_dir = Path('C:/_data/ai_factory/dataset')
train_img_dir = data_dir / 'train_img'
train_mask_dir = data_dir / 'train_mask'
test_img_dir = data_dir / 'test_img'
output_dir = data_dir / 'train_output'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

# Constants
MAX_PIXEL_VALUE = 65535
N_FILTERS = 16
N_CHANNELS = 3
EPOCHS = 330
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
MODEL_NAME = 'unet'
RANDOM_STATE = 312
INITIAL_EPOCH = 0
EARLY_STOP_PATIENCE = 10
CHECKPOINT_PERIOD = 5
CUDA_DEVICE = "0"

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
tf.config.set_soft_device_placement(True)
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except Exception as e:
    print(e)

train_meta = pd.read_csv(data_dir / 'train_meta.csv')
test_meta = pd.read_csv(data_dir / 'test_meta.csv')

# Define models and functions here...
# Make sure to replace the get_model function and other model-related code with the updated versions.

# Splitting data
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
images_train = [str(train_img_dir / image) for image in x_tr['train_img']]
masks_train = [str(train_mask_dir / mask) for mask in x_tr['train_mask']]
images_validation = [str(train_img_dir / image) for image in x_val['train_img']]
masks_validation = [str(train_mask_dir / mask) for mask in x_val['train_mask']]

from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=32, image_size=(256, 256), shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths = [self.image_paths[k] for k in indexes]
        mask_paths = [self.mask_paths[k] for k in indexes]
        X, y = self.__data_generation(image_paths, mask_paths)
        return X, y

    def __data_generation(self, image_paths, mask_paths):
        X = np.empty((self.batch_size, *self.image_size, N_CHANNELS))
        y = np.empty((self.batch_size, *self.image_size, 1), dtype=int)
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            img = rasterio.open(img_path).read().transpose((1, 2, 0))
            mask = rasterio.open(mask_path).read(1)

            img = np.float32(img) / MAX_PIXEL_VALUE
            mask = np.expand_dims(mask, axis=-1)

            X[i,] = img
            y[i,] = mask
        
        return X, y

def build_unet(input_shape=(256, 256, 3), n_filters=16, dropout=0.1, batchnorm=True):
    # Define the input
    inputs = Input(input_shape)
    # Contracting Path
    c1 = Conv2D(inputs, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    # Expansive Path
    # Complete the U-Net model architecture based on your specific requirements
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# You should complete the U-Net architecture as required for your task.
train_gen = DataGenerator(images_train, masks_train, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True)
val_gen = DataGenerator(images_validation, masks_validation, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False)

model = build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), n_filters=N_FILTERS, dropout=0.1, batchnorm=True)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_path = output_dir / f"{MODEL_NAME}_{save_name}_checkpoint.h5"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, es])

predictions = {}
for image_path in tqdm(test_img_dir.glob('*.tif')):
    img = rasterio.open(image_path).read().transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    pred_mask = model.predict(img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Apply threshold
    predictions[image_path.name] = pred_mask.squeeze()

# Save predictions as needed, potentially using joblib or numpy for arrays
prediction_output_path = output_dir / "predictions.pkl"
joblib.dump(predictions, prediction_output_path)
