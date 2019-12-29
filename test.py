from model import *
import cv2
import matplotlib.pyplot as plt
import copy
from utils import *

image_size = 128
chanels = 3

def test(test_image, weight_path = 'weight/unet.h5'):
    model = build_unet(image_size, chanels)
    model.load_weights(weight_path)
    image = cv2.imread(test_image)
    original_image = copy.deepcopy(image)
    image = cv2.resize(image, (image_size, image_size))
    predict = model.predict(image.reshape(-1, image_size, image_size, 3)/255.0)

    show_result(original_image, predict[0])

def show_result(image, result):
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (image_size, image_size)))
    ax1.title.set_text('Actual image')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Predict truth labels')
    result = cv2.resize(onehot2image(result), (image_size, image_size))
    ax2.imshow(result)
    plt.show()

if __name__ == '__main__':
    test("data/images/img/0cdf5b5d0ce1_01.jpg")