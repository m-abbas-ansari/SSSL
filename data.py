import json
import torch
from PIL import Image
from random import sample
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.nn.functional import pad


def plot(img, fix, dur):
    """
    Plot Image with fixations weighted by their durations.
    """
    fig = plt.figure()
    ax = plt.axes(xlim=(0, img.size[0]), ylim=(img.size[1], 0))
    im = ax.imshow(img)

    ax.scatter(fix[:,1],fix[:,0], color='cyan', edgecolors="blue", alpha=0.5, s=dur)
    for i in range(fix.shape[0]):
        ax.annotate(str(i+1), (fix[i,1]-7, fix[i,0]+7))
    plt.show()

def read_dataset(anno_dir, datasets, val_ratio = 0.2):
    """
    We split each dataset in datasets list into train and val
    """
    train_anno = {'fixation': [],
                  'duration': [],
                  'img_id': [],
                  'dva': [],
                  'img_size': []}
    
    val_anno = {'fixation': [],
                'duration': [],
                'img_id': [],
                'dva': [],
                'img_size': []}
    
    ret_anno = (train_anno, val_anno)
    
    for data in datasets:
        anno = {}
        with open(f"{anno_dir}/ANNOTATIONS/{data}.json", 'r') as f:
            anno = json.load(f)
        ims = list(anno.keys())
        val_len = int(val_ratio * len(ims))
        val_ims = sample(ims, val_len)
        train_ims = list(set(ims) - set(val_ims))
        
        for i, g in enumerate((train_ims, val_ims)):
            for im in g:
                num = len(anno[im]['fixations'])
                ret_anno[i]['fixation'].extend(anno[im]['fixations'])
                if 'durations' in anno[im]:
                    ret_anno[i]['duration'].extend(anno[im]['durations'])
                else:
                    dur = [[0]*len(fix) for fix in anno[im]['fixations']]
                    ret_anno[i]['duration'].extend(dur)
                    
                ret_anno[i]['img_id'].extend([f"{anno_dir}/STIMULI/{data}/{im}.jpg"]*num)
                if 'dva' in anno[im]:
                    ret_anno[i]['dva'].extend([anno[im]['dva']]*num)
                else:
                    ret_anno[i]['dva'].extend([None]*num)
                ret_anno[i]['img_size'].extend([anno[im]['img_size']]*num)
    
    return ret_anno

class FixationDataset(data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.data['img_id'][index]).convert('RGB')         
        fix = np.array(self.data['fixation'][index].copy())
        dur = np.array(self.data['duration'][index].copy())
        dva = self.data['dva'][index]
        
        v1, v2 = self.transform((img, fix, dur))
        
        return v1, v2
    
    def __len__(self):
        return len(self.data['img_id'])
    
    def collate_function(self, batch):
        v1_b, v2_b = zip(*batch)
        return self.pad_batch(v1_b), self.pad_batch(v2_b)

    def pad_batch(self, batch):
        img_b, fix_b, dur_b = zip(*batch)
        fix_b, dur_b = list(fix_b), list(dur_b)
        fix_lens = [len(fix) for fix in fix_b]
        max_fix_len = max(fix_lens)

        for i, fix in enumerate(fix_b):
            fix_b[i] = pad(fix, (0,0, 0, max_fix_len - len(fix)), "constant", 1e7)
            dur_b[i] = pad(dur_b[i], (0,max_fix_len - len(dur_b[i])), "constant", 1e7)

        return (torch.stack(img_b), torch.stack(fix_b), torch.stack(dur_b))