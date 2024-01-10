import torch
import random
import numpy as np
import cv2
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from copy import deepcopy

class ScanpathTransform:
    def __init__(self,
                 img_size = (320, 512),
                 noise = 0.6,
                 drop = 0.4,
                 reversal = 0.5,
                 rotation = 0.5,
                 ):
        
        self.transform = transforms.Compose([
            Resize(img_size),
            RandomHorizontalFlip(),
            transforms.RandomApply([ScanpathReversal()], p=reversal),
            transforms.RandomApply([Rotation(-30, 30)], p=rotation),
            transforms.RandomApply([FixNoiseAddition()], p=noise),
            transforms.RandomApply([FixDropout()], p=drop),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            Resize(img_size),
            transforms.RandomApply([FixNoiseAddition()], p=noise),
            transforms.RandomApply([FixDropout()], p=drop),
            RandomHorizontalFlip(),
            transforms.RandomApply([ScanpathReversal()], p=reversal),
            transforms.RandomApply([Rotation(-30, 30)], p=rotation),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        y1 = self.transform(deepcopy(x))
        y2 = self.transform(deepcopy(x))
        del x
        return y1, y2

class ResizeNormalize:
    def __init__(self,
                 img_size = (320, 512)
                 ):
        
        self.transform = transforms.Compose([
            Resize(img_size),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        img, fix, dur = self.transform(deepcopy(x))
        return img,fix,dur
    
class FixNoiseAddition(nn.Module):
    """
    Add random noise to each unnormalized fixation coordinate
    """
    def __init__(self, mean=0, std=25):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, object):
        img, fix, dur = object
        w, h = img.size
        noise = np.random.uniform(self.mean, self.std, size=(len(fix), 2))
        new_fix = np.add(fix, noise)
        for i,f in enumerate(new_fix):
            new_fix[i] = [max(0, min(h,f[0])), max(0, min(w,f[1]))]
        return (img, new_fix, dur)

class DurNoiseAddition(nn.Module):
    """
    Add random noise (in ms) to each duration
    """
    def __init__(self, mean=0, std=50):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, object):
        img, fix, dur = object
        noise = np.random.uniform(self.mean, self.std, size=(len(dur), 1))
        new_dur = np.add(dur, noise)
        
        return (img, fix, new_dur)
    
class RandomHorizontalFlip(nn.Module):
    """
    Performs random horizontal flipping of image along with fixations and durations
    """
    def __init__(self, p=0.4):
        super().__init__()
        self.p = p
        
    def forward(self, object):
        img, fix, dur = object
        if torch.rand(1) < self.p:
            w,_ = img.size
            img = TF.hflip(img)
            fix[:,1] = w - fix[:,1]
        return (img, fix, dur)

class FixDropout(nn.Module):
    """
    Drops fixations randomly
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        
    def forward(self, object):
        img, fix, dur = object
        if len(fix) > 2:
            new_fix = [fix[0]]
            new_dur = [dur[0]]
            for i, f in enumerate(fix):
                if i == 0:
                    continue
                if torch.rand(1) > self.p:
                    new_fix.append(f)
                    new_dur.append(dur[i])
        else:
            new_fix = fix
            new_dur = dur
            
        return (img, np.array(new_fix), np.array(new_dur))

class ScanpathReversal(nn.Module):
    """
    Reverses Scanpaths
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, object):
        img, fix, dur = object
        return (img, fix[::-1], dur[::-1])
    
class RotateOnlyImage(nn.Module):
    """
    Rotates image randomly, while fixations remain stationery
    """ 
    def __init__(self, l_angle, u_angle):
        super().__init__()
        self.l_angle = l_angle
        self.u_angle = u_angle
        
    def forward(self, object):
        img, fix, dur = object
        angle = random.randint(self.l_angle, self.u_angle)
        w,h = img.size
        (cX, cY) = (w // 2, h // 2) 
        new_img = img.rotate(angle, expand=True)
        nW, nH = new_img.size
        new_img = new_img.resize((w,h))
        new_fix = []
        for f in fix:
            new_fix.append([(f[0] + (nH / 2) - cY)*(h/nH), (f[1] + (nW / 2) - cX)*(w/nW)])
        
        return (new_img, np.array(new_fix), dur)

class FixPartialRotation(nn.Module):
    """
    Rotates image randomly, while fixations are rotated with an angle lesser than that of the image
    """
    def __init__(self, l_angle, u_angle):
        super().__init__()
        self.l_angle = l_angle
        self.u_angle = u_angle
        
    def forward(self, object):
        img, fix, dur = object
        angle = random.randint(self.l_angle, self.u_angle)
        fix_angle = random.randint(min(0, angle), max(0, angle))
        w,h = img.size
        (cX, cY) = (w // 2, h // 2) 
        new_img = img.rotate(angle, expand=True)
        nW, nH = new_img.size
        new_img = new_img.resize((w,h))
        
        M = cv2.getRotationMatrix2D((cX, cY), fix_angle, 1.0)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        new_fix = []
        for f in fix:
            v = [f[1],f[0],1]
            # Perform rotation
            calculated = np.dot(M,v)
            # Perfom resizing
            nY, nX = calculated[1] * (h/nH), calculated[0] * (w/nW)
            new_fix.append([nY, nX])
        return (new_img, np.array(new_fix), dur)
    
class FixFullRotation(nn.Module):
    """
    Rotates image along with the fixations together randomly
    """
    def __init__(self, l_angle, u_angle):
        super().__init__()
        self.l_angle = l_angle
        self.u_angle = u_angle
        
    def forward(self, object):
        img, fix, dur = object
        angle = random.randint(self.l_angle, self.u_angle)
        w,h = img.size
        (cX, cY) = (w // 2, h // 2) 
        new_img = img.rotate(angle, expand=True)
        nW, nH = new_img.size
        new_img = new_img.resize((w,h))
        
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        new_fix = []
        for f in fix:
            v = [f[1],f[0],1]
            # Perform rotation
            calculated = np.dot(M,v)
            # Perfom resizing
            nY, nX = calculated[1] * (h/nH), calculated[0] * (w/nW)
            new_fix.append([nY, nX])
        return (new_img, np.array(new_fix), dur)
    
class Rotation(nn.Module):
    """
    Picks one of the three rotation techniques randomly and applies it.
    1. Only Image Rotation
    2. Image and Fixation Rotation Together
    3. Image Rotation with Partial Fixation Rotation
    """
    def __init__(self, l_angle, u_angle):
        super().__init__()
        self.rotation_techniques = [RotateOnlyImage(l_angle, u_angle),
                                    FixFullRotation(l_angle, u_angle),
                                    FixPartialRotation(l_angle, u_angle)]
    
    def forward(self, object):
        technique_idx = random.randint(0,2)
        return self.rotation_techniques[technique_idx](object) 
    
class Resize(nn.Module):
    def __init__(self, new_size):
        super().__init__()
        self.new_size = new_size
        
    def forward(self, object):
        img, fix, dur = object
        nH, nW = self.new_size
        w, h = img.size
        new_img = img.resize((nW,nH))
        
        for i,f in enumerate(fix):
            fix[i] = [f[0] * (nH/h), f[1] * (nW/w)]
            
        return (new_img, fix, dur)

class ToTensor(nn.Module):
    def forward(self, object):
        img, fix, dur = object
        new_img = TF.to_tensor(img)
        new_fix = torch.from_numpy(fix.astype('float'))
        new_dur = torch.from_numpy(dur.astype('float'))
        
        return (new_img, new_fix, new_dur)        
    
class Normalize(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
        
    def forward(self, object):
        img, fix, dur = object
        _, h, w = img.size()
        new_img = TF.normalize(img, self.mean, self.std, self.inplace)
        fix[:, 0] /= h
        fix[:, 1] /= w
        fix = torch.clamp(fix, 0.0, 1.0)
        
        assert fix.max() <= 1.0, "Fixaitons must lie within image"
        
        return (new_img, fix, dur)
    






















