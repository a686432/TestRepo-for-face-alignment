from models import FAN
from utils import get_preds_fromhm, draw_gaussian
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from skimage import io
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    net = FAN()
    state_dict = torch.load('./checkpoint/dict.dp')
    net.load_state_dict(state_dict)
    net.cuda()
    img = Image.open("./test-input.jpg")
    width, height = img.width, img.height
    
    lms = []
    with open('test-target') as f:
        line = f.readline()
        line = line.split()
        tmp2 = []
        for i in line :
            tmp2.append(int(i))
            if(len(tmp2) == 2):
                tmp2.reverse()
                lms.append(tmp2)
                tmp2 = []
    f.close()
    heatmaps = []
    for pos in lms :
        heatmap = np.zeros((64, 64))
        if pos[0] >= 0 and pos[1] >= 0:
            heatmap = draw_gaussian(heatmap, pos, 1)
        heatmaps.append(heatmap)
    targets = torch.Tensor(heatmaps)

    targets = targets.cuda()
    targets = Variable(targets)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])
    criterion = nn.MSELoss()
    inputs = transform(img)
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2) )
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = net(inputs)
    loss = criterion(outputs[0][0][1], targets[1])
    print("LOSS:")
    print(loss)
    out = outputs[-1].data.cpu()[0]

# =============================================================================
#     center = torch.FloatTensor(
#                     [width / 2, height / 2])
#     center[1] = center[1] - (height) * 0.12
#     scale = (width + height) / 195.0
#     pts, pts_img = get_preds_fromhm(out, center, scale)
#     pts_img = pts_img.view(26, 2)
# =============================================================================
    
    
    pts_img = []
    for hm in out:
        yarray, yindex = torch.max(hm, 0)
        _, x = torch.max(yarray, 0)
        y = yindex[x]
        hm = hm.numpy()
#        print(hm.shape)
        if(hm[y, x] > 0.1):
            pts_img.append([int(y), int(x)])
    print(out[0])
#    print(heatmaps[0])
#    io.imshow(heatmaps[1] * 256 * 50)
#    io.imshow(out[1].numpy()+0.3)
        
    simg = io.imread("./test-input.jpg")

    for pos in pts_img:
        if pos[0] > 0 and pos[1] > 0:
            i = round(pos[0] * height / 64)
            j = round(pos[1] * width / 64)
            simg[int(i)-1:int(i)+1,int(j-1):int(j+1)]= [255, 0, 0]
#    io.imshow(simg)
