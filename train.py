from custom_data_generator import *
from model import *
from tensorflow.keras.optimizers import Adam
import os

image_size = 128
chanels = 3
epochs = 5
train_path = "data/images/img/"
batch_size = 10
validation_split = 0.2

def train():
    model = build_unet(image_size, chanels)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=['accuracy'])

    num_validation =  validation_split * len(os.listdir(train_path))
    num_training =  len(os.listdir(train_path)) - num_validation
    steps_per_epoch = num_training // batch_size
    validation_steps = num_validation // batch_size
    model.fit_generator(CustomGenerator("training", batch_size, validation_split, num_classes=1), validation_data=CustomGenerator("validation", batch_size, validation_split, num_classes=1), steps_per_epoch=steps_per_epoch,
                        epochs=epochs, validation_steps=validation_steps)
    model.save_weights("weight/unet.h5")

if __name__ == '__main__':
    train()