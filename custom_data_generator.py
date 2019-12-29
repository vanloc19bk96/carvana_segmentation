from utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def CustomGenerator(subset, batch_size, validation_split, num_classes, target_size=(128, 128), seed = 1):
    image_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    mask_datagen = ImageDataGenerator(
        validation_split=validation_split
    )

    image_generator = image_datagen.flow_from_directory(
        'data/images', batch_size=batch_size, seed=seed, subset=subset, target_size=target_size
    )

    mask_generator = mask_datagen.flow_from_directory(
        'data/masks', batch_size=batch_size, seed=seed, subset=subset, target_size=target_size, color_mode = 'grayscale'
    )
    while(True):
        images = image_generator.next()
        masks = mask_generator.next()
        encoded = [image2onehot(masks[0][x, :, :, :], num_classes) for x in range(masks[0].shape[0])]
        images = np.array([image for image in images[0]])
        yield images, np.asarray(encoded)

