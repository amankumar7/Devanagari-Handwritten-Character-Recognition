import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import tensorflow as tf


def data(flag):

    base_dir='D:\\Notes\\Python\\datasets\\DevanagariHandwrittenCharacterDataset\\DevanagariHandwrittenCharacterDataset'
    train_dir=os.path.join(base_dir,"Train")
    test_dir=os.path.join(base_dir,"Test")
    current_id=0
    labels_id={}
    img_size=32
    data=[]
    if flag.lower() == 'train' :
        dirt = train_dir
    if flag.lower() == 'test' :
        dirt = test_dir
    for root,dirs,files in os.walk(dirt):
        for file in tqdm(files):
            if file.endswith("png"):
                path=os.path.join(root,file)
                name,ext=os.path.splitext(os.path.basename(root).lower())
                label=name.split("_")[-1]
                if not label in labels_id:
                    labels_id[label]=current_id
                    current_id+=1
                id=labels_id[label]
                image_array=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                image_array=cv2.resize(image_array,(img_size,img_size))
                image_array=np.array(image_array)
                data.append([image_array,id])
    random.shuffle(data)
    x_data=[]
    y_data=[]
    for f,l in data:
        x_data.append(f)
        y_data.append(l)
    return (x_data, y_data , labels_id )
