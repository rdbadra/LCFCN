from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread
from skimage.transform import resize
from scipy.io import loadmat
import torchvision.transforms.functional as FT
from haven import haven_utils as hu
from . import transformers
from PIL import Image

from os import walk
import torch



class MicroPlastics(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        self.split = split
        self.exp_dict = exp_dict
        
        self.n_classes = 1
        IMAGE_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/images/'

        all_files = []
        for (dirpath, dirnames, filenames) in walk(IMAGE_DIR):
            all_files.extend(filenames)
            break

        self.img_names = [file.split('_dots.png')[0] for file in all_files if '_dots.png' in file]

        if split == "train":
            self.img_names = self.img_names[:int(len(self.img_names)*0.8)]
            print(f'TRAIN IMAGES: {len(self.img_names)}')

        elif split == "val":
            self.img_names = self.img_names[int(len(self.img_names)*0.8):]
            print(f'VALID IMAGES: {len(self.img_names)}')

        elif split == "test":
            #fname = os.path.join(datadir, 'image_sets', 'test.txt')
            pass


        #self.img_names = [name.replace(".jpg\n","") for name in hu.read_text(fname)]
        #self.img_names = [self.img_names[i] for i in range(len(self.img_names)) if i not in [334]]
        self.path = IMAGE_DIR
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        torch.cuda.empty_cache()
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))


        points = imread(os.path.join(self.path, name + "_dots.png"))[:,:,:1].clip(0,1)
        
        #points = resize(points, (224, 224),
        #               anti_aliasing=True)[:,:,:1].clip(0,1).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        #collection = list(map(FT.to_pil_image, [image, points]))
        # ADDED FROM HERE

        '''
        image, points = list(map(FT.to_pil_image, [image, points]))

        image = np.array(image.getdata())
        points = np.array(points.getdata())

        print(f'SUM: {np.array([int(points.sum())])}')
        print(f'Squeeze: {points.squeeze()}')
        '''

        # UNTIL HERE

        
        image, points = transformers.apply_transform(self.split, image.copy(), points, 
                   transform_name=self.exp_dict['dataset']['transform'])

        # print(f'Image: {index} - {image.shape}')

        
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}
