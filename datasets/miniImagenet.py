##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm

import numpy as np
np.random.seed(2191)  # for reproducibility

# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)
PiLImageResize = lambda x: x.resize((84,84))

class DatasetLoader(data.Dataset):
    def __init__(self, dataroot = './data/miniImagenet', type = 'train',
                 nEpisodes = 1000, classes_per_set=10, samples_per_class=1):

        self.nEpisodes = nEpisodes
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.samples_per_class_eval = 15
        self.n_samples = self.samples_per_class * self.classes_per_set
        self.n_samples_eval = self.samples_per_class_eval * self.classes_per_set
        # Transformations to the image
        self.transform = transforms.Compose([filenameToPILImage,
                                             PiLImageResize,
                                             transforms.ToTensor()
                                             ])

        def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels

        #requiredFiles = ['train','val','test']
        self.miniImagenetImagesDir = os.path.join(dataroot,'images')
        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.classes_dict = {self.data.keys()[i]:i  for i in range(len(self.data.keys()))}
        self.create_episodes(self.nEpisodes)

    def create_episodes(self,episodes):

        nClasses = len(self.data.keys())

        self.support_set_x_batch = []
        self.target_x_batch = []
        for b in np.arange(episodes):
            # select n classes_per_set randomly
            selected_classes = np.random.choice(nClasses, self.classes_per_set, False)
            support_set_x = []
            target_x = []
            for c in selected_classes:
                selected_samples = np.random.choice(len(self.data[self.data.keys()[c]]),
                               self.samples_per_class + self.samples_per_class_eval, False)
                indexDtrain = np.array(selected_samples[:self.samples_per_class])
                indexDtest = np.array(selected_samples[self.samples_per_class:])
                support_set_x.append(np.array(self.data[self.data.keys()[c]])[indexDtrain].tolist())
                target_x.append(np.array(self.data[self.data.keys()[c]])[indexDtest].tolist())
            self.support_set_x_batch.append(support_set_x)
            self.target_x_batch.append(target_x)

    def __getitem__(self, index):

        support_set_x = torch.FloatTensor(self.n_samples, 3, 84, 84)
        support_set_y = np.zeros((self.n_samples), dtype=np.int)
        target_x = torch.FloatTensor(self.n_samples_eval, 3, 84, 84)
        target_y = np.zeros((self.n_samples_eval), dtype=np.int)

        flatten_support_set_x_batch = [os.path.join(self.miniImagenetImagesDir,item)
                                       for sublist in self.support_set_x_batch[index] for item in sublist]
        support_set_y = np.array([self.classes_dict[item[:9]]
                                      for sublist in self.support_set_x_batch[index] for item in sublist])
        flatten_target_x = [os.path.join(self.miniImagenetImagesDir,item)
                            for sublist in self.target_x_batch[index] for item in sublist]
        target_y = np.array([self.classes_dict[item[:9]]
                            for sublist in self.target_x_batch[index] for item in sublist])

        for i,path in enumerate(flatten_support_set_x_batch):
            if self.transform is not None:
                support_set_x[i] = self.transform(path)

        for i,path in enumerate(flatten_target_x):
            if self.transform is not None:
                target_x[i] = self.transform(path)
        return support_set_x, torch.IntTensor(support_set_y), target_x, torch.IntTensor(target_y)

    def __len__(self):
        return self.nEpisodes


