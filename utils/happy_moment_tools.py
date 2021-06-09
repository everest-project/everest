import os, sys
import numpy as np
import torch
import torchvision.transforms as t

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm


from PIL import Image

  
    
class ImageDataset (Dataset):

    def __init__(self, imgs,  transform=None):
        super(ImageDataset).__init__()
    
        self.imgs = imgs
        self.transform = transform
        
    def __getitem__(self, index):
        img =Image.fromarray(self.imgs[index])
        
        if self.transform:
            x = self.transform(img)
        
        return x
    
    def __len__(self):
        return len(self.imgs)
    
# pretrianed_models = ('hybrid_finetuned_fc6+','hybrid_finetuned_all','vgg19_finetuned_fc6+', 'vgg19_finetuned_all')

def sentiment_analysis(cropped_imgs,model, batch_size=8):

    transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Lambda(lambda x: x[[2,1,0], ...] * 255),  # RGB -> BGR and [0,1] -> [0,255]
        t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1,1,1]),  # mean subtraction
    ])

    
    data = ImageDataset(cropped_imgs,  transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=0, pin_memory=True)
    

    score = []
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            p = model(x.to('cuda')).cpu().numpy()  # order is (NEG, NEU, POS)
            for single_pic in p:
                score.append( single_pic[2]-single_pic[0])

    return score
    
