# Modified from https://github.com/cvlab-stonybrook/Target-absent-Human-Attention/blob/main/irl_ffm/models.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def get_foveal_weights(fixation_batch,
                       width,
                       height,
                       sigma=0.248,
                       p=7.5,
                       k=1.5,
                       alpha=1.25):
    """
    This function generate foveated image in batch on GPU
    
    fixation_batch: normalized fixation tensor of shape (batch_size, 
      fix_num, 2) 
    """
    assert fixation_batch.size(-1) == 2, 'Wrong input shape!'
#     assert fixation_batch.max() <= 1, 'Fixation has to be normalized!'
    prNum = 5

    batch_size = fixation_batch.size(0)
    fix_num = fixation_batch.size(1)
    device = fixation_batch.device

    # Map fixations to coordinate space
    fixation_batch = fixation_batch * torch.tensor([width, height]).to(device)

    x = torch.arange(0, width, device=device, dtype=torch.float)
    y = torch.arange(0, height, device=device, dtype=torch.float)
    y2d, x2d = torch.meshgrid([y, x])
    h, w = x2d.size()

    x2d = x2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)
    y2d = y2d.view(1, 1, h, w).expand(batch_size, fix_num, -1, -1)

    # fixation patch index to fixation pixel coordinates
    xc = fixation_batch[:, :, 0]
    yc = fixation_batch[:, :, 1]

    xc2d = xc.view(batch_size, fix_num, 1, 1).expand_as(x2d)
    yc2d = yc.view(batch_size, fix_num, 1, 1).expand_as(y2d)

    theta = torch.sqrt((x2d - xc2d)**2 + (y2d - yc2d)**2) / p
    theta, _ = torch.min(theta, dim=1)
    R = alpha / (theta + alpha)

    Ts = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum - 1):
        Ts[:, i] = torch.exp(-((2**(i - 2)) * R / sigma)**2 * k)

    # omega
    omega = torch.zeros(prNum)
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)
    omega[:-1] = torch.sqrt(math.log(2) / k) / (2**torch.arange(
        -2, prNum // 2, dtype=torch.float32, device=device)) * sigma
    omega[omega > 1] = 1

    # layer index
    layer_ind = torch.zeros_like(R, device=device)
    for i in range(1, prNum):
        ind = (R >= omega[i]) * (R <= omega[i - 1])
        layer_ind[ind] = i

    # Bs
    Bs = (0.5 - Ts[:, 1:]) / (Ts[:, :-1] - Ts[:, 1:])

    # M
    Ms = torch.zeros((batch_size, prNum, height, width), device=device)
    for i in range(prNum):
        ind = layer_ind == i
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[:, i][ind] = 1
            else:
                Ms[:, i][ind] = 1 - Bs[:, i - 1][ind]

        ind = layer_ind - 1 == i
        if torch.sum(ind) > 0:
            Ms[:, i][ind] = Bs[:, i][ind]

    return Ms
    
class FeatureFovealNet(nn.Module):
    def __init__(self,
                 feat_dim):
        super(FeatureFovealNet, self).__init__()

        self.feat_dim = feat_dim
        self.out_dims = [256, 512, 1024, 2048]
        self.conv1x1_ops = []
        for dim in self.out_dims:
            self.conv1x1_ops.append(nn.Conv2d(dim, self.feat_dim, 1))
        self.conv1x1_ops = nn.ModuleList(self.conv1x1_ops)

#         self.fovea_size = nn.Parameter(torch.tensor(4))
        self.amplitude = nn.Parameter(torch.tensor(0.68))
        self.acuity = nn.Parameter(torch.tensor(1.25))
        

    def forward(self, conv_outs, fixations, get_weight=True, is_cumulative=True):

        uni_conv_outs = [conv_outs[0]]
        for i, out in enumerate(conv_outs[1:]):
            uni_conv_outs.append(F.upsample(out, scale_factor=2**(i + 1)))

        for i in range(len(uni_conv_outs)):
            uni_conv_outs[i] = self.conv1x1_ops[i](
                uni_conv_outs[i]).unsqueeze(1)

        uni_conv_outs = torch.cat(uni_conv_outs, dim=1)

        h, w = uni_conv_outs.shape[-2:]

        weights = get_foveal_weights(
            fixations,
            w,
            h,
            p=4, #self.fovea_size,
            k=self.amplitude,
            alpha=self.acuity) if get_weight else fixations
        weights = weights[:, :-1]
        weights = weights.unsqueeze(2).expand(-1, -1, self.feat_dim, -1, -1)
        foveal_features = torch.sum(weights * uni_conv_outs, dim=1)

        return foveal_features

class FFNGenerator(nn.Module):
    def __init__(self,
                 img_size,
                 foveal_feat_size=256,
                 is_cumulative=True, 
                 ):
        super(FFNGenerator, self).__init__()
        self.is_cumulative = is_cumulative
        self.FFN = FeatureFovealNet(foveal_feat_size)
#         hidden_size = foveal_feat_size
        self.resnet = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        hidden_size = foveal_feat_size * 2
        H, W = img_size
        self.layer2 = nn.Sequential(
            nn.Conv2d(foveal_feat_size,
                      hidden_size,
                      3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.LayerNorm([hidden_size, H//4, W//4]), nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 1, stride=2, bias=False),
            nn.LayerNorm([hidden_size, H//8, W//8]))

        hidden_size *= 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(foveal_feat_size*2,
                          hidden_size,
                          3,
                          stride=2,
                          padding=1,
                          bias=False), 
            nn.LayerNorm([hidden_size, H//16, W//16]), nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 1, stride=2, bias=False),
            nn.LayerNorm([hidden_size, H//32, W//32]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        5
        
    def forward(self, view):
        im_tensor, fixations, _ = view
        # Construct foveated feature from images and fixations as state representation
        conv1_out = self.resnet.conv1(im_tensor)
        x = self.resnet.bn1(conv1_out)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        conv2_out = self.resnet.layer1(x)
        conv3_out = self.resnet.layer2(conv2_out)
        conv4_out = self.resnet.layer3(conv3_out)
        conv5_out = self.resnet.layer4(conv4_out)
        conv_outs = [conv2_out, conv3_out, conv4_out, conv5_out]

        x = self.FFN(conv_outs, fixations, is_cumulative=self.is_cumulative)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        return out
    
