import os
import pickle
import pandas as pd
import cv2
import random
import numpy as np
import util
import multiprocessing as mp


class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, image_path, label_path, selected_attributes, batch_size, mode='train'):
        super(DataLoader, self).__init__()
        self.image_path = image_path
        self.batch_size = batch_size
        self.selected_attributes = selected_attributes

        with open(os.path.join(label_path, 'x2y.pkl'), 'rb') as f:
            self.x2y = pickle.load(f)
        with open(os.path.join(label_path, 'attr2idx.pkl'), 'rb') as f:
            self.attr2idx = pickle.load(f)

        if mode == 'train':
            self.image_files = self.x2y.keys()[2000:]
        elif mode == 'test':
            self.image_files = self.x2y.keys()[:2000]
        else:
            raise ValueError('Mode can only be train or test...')

        self.total_len = len(self.image_files)
        self.num_batches = int(self.total_len / batch_size)
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self


    def reset(self):
        self.idx = 0
        random.shuffle(self.image_files)


    def __next__(self):
        return self.next()

    
    def get_img(self, image_name):


        img_bgr = cv2.imread(os.path.join(self.image_path, image_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # img_rgb_normalize = self.preprocess(img_rgb)

        return img_rgb

    def preprocess(self, img):

        img = img / 255.
        return img

    def get_attr(self, image_name):

        label = []
        for attr in self.selected_attributes:
            label.append(self.x2y[image_name][self.attr2idx[attr]])

        return label

    def next(self):
        
        if self.idx >= self.total_len:
            raise StopIteration

        image_names = self.image_files[self.idx: self.idx + self.batch_size]
        images = np.array([self.get_img(name) for name in image_names])
        labels = np.array([self.get_attr(name) for name in image_names])
        self.idx += self.batch_size

        return images, labels  # (batch_size, 218, 178, 3), (None, '#' of selected_attr)


## unit test ##
if __name__ == '__main__':

    IMAGE_PATH = './data/CelebA_nocrop/images'
    LABEL_PATH = './data/'
    attributes = ['5_o_Clock_Shadow','Arched_Eyebrows',
                'Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair',
                'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair',
                'Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face',
                'Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
                'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']

    dataloader = DataLoader(IMAGE_PATH, LABEL_PATH, attributes[0:3], 100, mode='test')
    X, y = dataloader.__iter__().next()
    print (y)
    






        
