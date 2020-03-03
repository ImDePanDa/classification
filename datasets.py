import os 
import pandas
import cv2 
import torch
import numpy as np 

class csvdataset():
    def __init__(self, root, csvnames):
        self.root = root
        self.imgnames = []
        self.labels = []
        for csvname in csvnames:
            csvname = csvname+".csv"
            csvpath = os.path.join(root, csvname)
            csvInfo = pandas.read_csv(csvpath)
            self.imgnames += list(map(lambda x: csvInfo.loc[x].values[0], range(len(csvInfo))))
            self.labels += list(map(lambda x: csvInfo.loc[x].values[1], range(len(csvInfo))))

    def __getitem__(self, index):
        imgPath = os.path.join(self.root, "train", str(self.labels[index]), self.imgnames[index])
        # print(imgPath)
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)