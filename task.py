from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.datasets import cifar100
from keras.preprocessing import image
import imageio
import numpy as np
import random

#Download glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip
class Task:
    def __init__(self, batch_size):
        self.classes = np.array(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])
        self.batch_size = batch_size
        self.w_img_pre = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.w_txt_pre = dict()
        f = open('../datasets/glove/glove.6B.50d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.w_txt_pre[word] = coefs
        f.close()
        train, test = cifar100.load_data(label_mode='fine')
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test
        self.tr_i = 0

    def train_batch(self):
        batch_x = self.x_train[self.tr_i:self.tr_i + self.batch_size]
        batch_y = self.y_train[self.tr_i:self.tr_i + self.batch_size]
        imgs = np.zeros((self.batch_size, 224, 224, 3))
        txt_pre = np.zeros((self.batch_size, 50,))

        for i, x in enumerate(batch_x):
            img = image.array_to_img(x)
            img = img.resize((224, 224))
            imgs[i] = image.img_to_array(img)

        img_pre = preprocess_input(imgs)
        img_pre = self.w_img_pre.predict(img_pre)

        classes_neg = random.sample(list(self.classes), int(self.batch_size / 2))
        classes_pos = self.classes[batch_y][:int(self.batch_size / 2)].reshape(-1,)
        classes = np.concatenate((classes_neg, classes_pos))
        for i, one_class in enumerate(classes):
            if one_class not in ['lawn_mower', 'aquarium_fish', 'maple_tree', 'willow_tree', 'palm_tree', 'oak_tree', 'pickup_truck', 'pine_tree', 'sweet_pepper']:
                txt_pre[i] = self.w_txt_pre[one_class]

        ys = [1] * (self.batch_size // 2) + [-1] * (self.batch_size // 2)

        return img_pre, txt_pre, ys
