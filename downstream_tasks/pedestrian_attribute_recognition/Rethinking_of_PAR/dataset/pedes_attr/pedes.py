import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import random
from tools.function import get_pkl_rootpath
import torch

class PedesAttr(data.Dataset):

    def __init__(self, cfg, split, transform=None, target_transform=None, idx=None):

        assert cfg.DATASET.NAME in ['PETA','EventPAR','Mars','DUKE', 'PA100k', 'RAP', 'RAP2', 'MSPAR'], \
            f'dataset name {cfg.DATASET.NAME} is not exist'

        #data_path = get_pkl_rootpath(cfg.DATASET.NAME, cfg.DATASET.ZERO_SHOT)

        #print("which pickle", data_path)

        # dataset_info = pickle.load(open('/data/jinjiandong/dataset/MSP_degrade/dataset_random.pkl', 'rb+'))
        #cross_domain pkl加载
        #dataset_info = pickle.load(open("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/annotation/dataset_reorder.pkl",'rb+')  ) 
        dataset_info = pickle.load(open("/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/DUKE/DUKE_pre/duke_annotation/pad_duke.pkl",'rb+')  )

        img_id = dataset_info.track_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        
        self.transform = transform
        self.target_transform = target_transform
        self.attributes = dataset_info.attr_name
        self.root_path ="/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/DUKE/pad_duke_dataset"
        #self.root_path ='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/EventPAR'
        print(self.root_path)
        self.attr_num = len(self.attributes)
        self.eval_attr_num=len(self.attributes)
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label
        
        self.label_vector = dataset_info.attr_vectors
        self.label_word = np.array(dataset_info.attr_name)

        self.words = self.label_word.tolist()
        self.result_dict=dataset_info.result_dict

    def __getitem__(self, index):

        imgname= self.img_id[index]
        gt_label = self.result_dict[imgname]

        rgb_imgpath = os.path.join(self.root_path, imgname)
        event_imgpath =rgb_imgpath.replace(imgname,imgname+'_event')
        rgb_img_list=sorted(os.listdir(rgb_imgpath))
       
        rgb_list=[]
        event_list=[]

        for file in random.choices(rgb_img_list,k=6):
            rgb_file = os.path.join(rgb_imgpath, file)
            event_file = os.path.join(event_imgpath, file)
            rgb_pil = Image.open(rgb_file)
            event_pil = Image.open(event_file)
            if self.transform is not None:
                rgb_pil = self.transform(rgb_pil)
                event_pil = self.transform(event_pil)
            rgb_list.append(rgb_pil)
            event_list.append(event_pil)
        rgb_img = torch.stack(rgb_list, dim=0)
        event_img = torch.stack(event_list, dim=0)
      

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return rgb_img,event_img, gt_label, imgname,  # noisy_weight

    def __len__(self):
        return len(self.img_id)

