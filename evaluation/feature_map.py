import os
from ResNet import ResNet18
import PIL.Image as Image
import torch
import numpy as np
from torchvision.utils import save_image


if os.path.exists('./model/evaluation-and-plot - breast/evaluation_and_plot-91feature_maps/features-label0.png') is not True:
    net = ResNet18()
    torch.load('./net200.pkl', map_location='cuda:0')
    path = './features'
    files = []
    for _, _, file in os.walk(path):
        if file:
            files.append(file)
    for j, data in enumerate(files[0]):
        inputs = Image.open(os.path.join(path, data)).convert('RGB')
        inputs = np.asarray(inputs)
        inputs = torch.tensor(inputs)
        inputs = inputs.permute(2, 0, 1)
        inputs = inputs.unsqueeze(0).float()
        output = net.conv1(inputs)  # 提取第一层卷积层的特征
        new_output = output[0]
        img = inputs[0]
        new_output = new_output.data
        for i in range(64):
            save_image(new_output[i], os.path.join(path, '{}'.format(data[:-4]), '{}.png'.format(i)))
            print('finished get features-label{}.png'.format(j))



