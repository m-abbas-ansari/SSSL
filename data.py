import os
import glob
import json
import shutil
from PIL import Image
from random import sample

def read_dataset(datasets, val_ratio = 0.2):
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
        with open(f"ANNOTATIONS/{data}.json", 'r') as f:
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
                    ret_anno[i]['duration'].extend([0]*num)
                    
                ret_anno[i]['img_id'].extend([f"STIMULI/{data}/{im}.jpg"]*num)
                ret_anno[i]['dva'].extend([anno[im]['dva']]*num)
                ret_anno[i]['img_size'].extend([anno[im]['img_size']]*num)
    
    return ret_anno

class FixDataset(data.Dataset):
    def __init__(self, data, max_len, img_height, img_width, transform):
        self.fixation = data['fixation']
        self.duration = data['duration']
        
        self.img_id = data['img_id']
        self.dva = data['dva']
        self.img_size = data['img_size']
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(self.img_id[index])
        if self.transform is not None:
            img = self.transform(img)
            
        fix = self.fixation[index]
        dur = self.duration[index]
        dva = self.dva[index]
        img_size = self.img_size[index]

        
        while len(fix) < self.max_len:
            fix.append(0)
            dur.append(0)
            
        fixation = torch.from_numpy(np.array(fix[:self.max_len]).astype('int'))
        duration = torch.from_numpy(np.array(dur[:self.max_len]).astype('int'))
        
        # Work in PROGRESS