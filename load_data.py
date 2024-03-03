import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms as T




class ImageDataset (Dataset):
    def __init__(self, path_file, split = 'train', transform = None):
        self.root_dir = os.path.join(path_file, split)  # assuming that the data is in a folder called "images"
        self.transform = transform
        
        self.img_file = []
        self.img_classes = []
        self.img_labels = []
        for rooth_dir, subdir, filenames in os.walk(self.root_dir):
            for dirs in subdir:
                self.img_classes.append(dirs)
            for files in filenames:
                img_path = os.path.join(rooth_dir, files)
                self.img_file.append(img_path)
                self.img_labels.append(self.img_classes.index(img_path.split(os.sep)[-2]))  # get the class of image
            
    
    def __len__ (self, ):
        return len(self.img_file)


    def __getitem__ (self, idx):
        single_image = Image.open(self.img_file[idx]).convert('RGB')
        single_label = self.img_labels[idx]
        
        if self.transform is not None:
            single_image = self.transform(single_image)
        else:
            single_image = T.ToTensor()(single_image)
            
        return single_image, single_label





class Image_from_csv(Dataset):
    def __init__ (self, root_dir, csv_file, split = 'train',transform = None):
        os.chdir(root_dir)
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.split = split 
        self.transform = transform
        
        self.img_file = []
        self.img_classes = []
        self.img_labels = []
        
        self.data = pd.read_csv(csv_file)
        
        data = self.data.query('`data set` == @self.split')
        self.img_file.extend(data['filepaths'].values.tolist())
        self.img_labels.extend([int(x) for x in data['class id'].values])
        self.img_classes.extend(sorted(set([x for x in data['labels'].values]), reverse = False))
        
    
    def __len_ (self):
        return len(self.img_file)
    
    def __getitem__ (self, idx):
        
        single_image = Image.open(self.img_file[idx]).convert('RGB')
        single_label = self.img_labels[idx]
        
        if self.transform is not None:
            single_image = self.transform(single_image)
        else:
            single_image = T.ToTensor()(single_image)
            
        return single_image, single_label
            