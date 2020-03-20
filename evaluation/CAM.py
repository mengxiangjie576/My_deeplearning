import io
import requests
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
from ResNet import ResNet18

# so CAM could be used directly.
# net = models.squeezenet1_1(pretrained=False)
net = ResNet18()
torch.load('./model/net.pkl')
finalconv_name = 'layer3'  # this is the last conv layer of the network

net.eval()

# hook the feature extractor
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net._modules.get(finalconv_name).register_forward_hook(hook_feature)


# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-3].data.numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    normalize
])

img_pil = Image.open('./model/evaluation_and_plot/feature_maps/img-label0.jpg')
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil).unsqueeze(0)
logit = net(img_tensor)


h_x = F.softmax(logit[0], dim=0).data.squeeze()
probs, idx = h_x.sort(0, True)
idx = idx.numpy()


# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)



