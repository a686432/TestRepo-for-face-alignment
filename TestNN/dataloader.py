import os, shutil
import numpy as np
import torch, torch.nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms
from torch.autograd import Variable
from PIL import Image
from models import FAN
from utils import draw_gaussian

def retLandmarks(landmarkfile):
    results = None
    with open(landmarkfile) as f:
        lines = f.readlines()
        results = np.ones((26, 2))
        results = results * -1
        for line in lines:
            landmark = line.split()
            # print(str(landmark) +'\n')
            results[int(landmark[0])] = [int(landmark[1]), int(landmark[2])]
    return results



def convert_to_data(train=True):
    basedir = r'../'
    trainsrcdir = r'/home/kg/face-alignment/depth_img_uint8/train/'
    testsrcdir = r'/home/kg/face-alignment/depth_img_uint8/test/'
    landmarkdir = r'/home/kg/face-alignment/landmark_2D/'
    if not os.path.exists(trainsrcdir):
        return

    if(train):
        f = open(basedir + 'train.txt', 'w')
        train_path = basedir + 'train/'
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        
        dirs = os.listdir(trainsrcdir)
        for dir in dirs:
            srcfiles = os.listdir(trainsrcdir + dir)
            for srcimg in srcfiles:
                print(srcimg + '\n')
                shutil.copyfile(trainsrcdir + dir + '/' + srcimg, train_path + srcimg)

                imgname = train_path + srcimg
                img = Image.open(imgname)
                width, height = img.width, img.height

                lmdir = 'ld' + dir[2:]
                landmarkfile = landmarkdir + lmdir + '/' + srcimg[:-3] + 'txt'
                landmarks = retLandmarks(landmarkfile)
                f.write(train_path + srcimg + ' ')
                for lm in landmarks:
                    h = round(int(lm[0]) * 256 / height) if int(lm[0]) > 0 else int(lm[0])
                    w = round(int(lm[1]) * 256 / width) if int(lm[1]) > 0 else int(lm[1])
                    f.write(str(int(h)) + ' ' + str(int(w)) + ' ')
                f.write('\n')
        f.close()
    else:
        f = open(basedir + 'test.txt', 'w')
        test_path = basedir + 'test/'
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        
        dirs = os.listdir(testsrcdir)
        for dir in dirs:
            srcfiles = os.listdir(testsrcdir + dir)
            for srcimg in srcfiles:
                shutil.copyfile(testsrcdir + dir + '/' + srcimg, test_path + srcimg)

                imgname = test_path + srcimg
                img = Image.open(imgname)
                width, height = img.width, img.height

                lmdir = 'ld' + dir[2:]
                landmarkfile = landmarkdir + lmdir + '/' + srcimg[:-3] + 'txt'
                landmarks = retLandmarks(landmarkfile)
                f.write(test_path + srcimg + ' ')
                for lm in landmarks:
                    h = (int(lm[0]) * 256 / height) if int(lm[0]) > 0 else int(lm[0])
                    w = (int(lm[1]) * 256 / width)if int(lm[1]) > 0 else int(lm[1])
                    f.write(str(int(h)) + ' ' + str(int(w)) + ' ')
                f.write('\n')
        f.close()

class MyDepthDataSet(Dataset):
    def __init__(self, train=True, root="../", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.train = train
        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.catalog = root + "train.txt"
        else:
            self.catalog = root + "test.txt"
        
        if not os.path.exists(self.catalog):
            convert_to_data(train=self.train)

        with open(self.catalog) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                lms = []
                tmp2 = []
                imgname = line[0]

                for i in line[1:] :
                    tmp2.append(int(i))
                    if(len(tmp2) == 2):
                        tmp2.reverse()
                        heatmap = np.zeros((256,256))
                        if tmp2[0] >= 0 and tmp2[1] >= 0:
                            heatmap = draw_gaussian(heatmap, tmp2, 1)
                        lms.append(heatmap)
                        tmp2 = []
                self.data.append([imgname, lms])
            f.close()


    def __getitem__(self, index):
        imgname, target = self.data[index]
        img = Image.open(imgname)
        width, height = img.width, img.height
        if self.transform is not None:
            img = self.transform(img)
        # if no rotation happens
        heatmaps = target
        heatmaps = torch.Tensor(heatmaps)
        return img, heatmaps

    def __len__(self):
        return len(self.data)

    

def Test():
    print("For test.\n")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])
    dataset = MyDepthDataSet(train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    print(dataset.__len__())
    inputdata, target = dataset.__getitem__(0)
    # print(inputdata)
    print(target.shape)
    # fan = FAN()
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     output = fan(Variable(inputs))
    #     print(output[-1].size())
    #     break
    

if __name__ == "__main__":
    convert_to_data(True)
    convert_to_data(False)