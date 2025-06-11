import os
import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import json
from torch.nn import functional as F
from typing import Type
from torch import Tensor
import math
from typing import Tuple
from data_preprocess._02_color_normalization import normalize_staining
from extract_features.scist import Sc_Guide_Model_with_CNNSASM
from extract_features.pps import CNN_SelfAttention_softmax

def load_scist():
    wts_path='./liver_cancer-ST_Net-Sc_Guide_Model_with_CNN_SASM-best.pth'
    scist = Sc_Guide_Model_with_CNNSASM('resnet34', 183)
    scist.load_state_dict(torch.load(wts_path))
    scist.eval()
    return scist

class Extract():
    def __init__(self,patch_path,coord_path,orig_exp_path,save_path,gpu=True):
        self.tcga_list=os.listdir(patch_path)
        self.scist=load_scist()
        self.gpu=gpu
        if gpu:
            self.scist.to('cuda:0')
        self.patch_path=patch_path
        self.coord_path=coord_path
        self.save_path=save_path
        self.orig_exp=orig_exp_path

    def extract_single_feature(self,svs_name):
        assert svs_name in self.tcga_list
        single_patch_path=os.path.join(self.patch_path,svs_name)
        feature_save_path=os.path.join(self.save_path,svs_name)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)
        for single_patch in os.listdir(single_patch_path):
            single_feature_dict={}
            spec_patch_path=os.path.join(single_patch_path,single_patch)
            scist_patch=self.scist_preprocessing(spec_patch_path)
            spec_orig_exp_path=os.path.join(self.orig_exp,svs_name,single_patch)
            if not os.path.exists(spec_orig_exp_path.replace('.png','.npy')):
                continue
            orig_exp=torch.Tensor(np.load(spec_orig_exp_path.replace('.png','.npy'))).squeeze()
            if self.gpu:
                scist_patch=scist_patch.to('cuda:0')
                orig_exp=orig_exp.to('cuda:0')
            single_exp=self.scist(scist_patch.unsqueeze(0),orig_exp.unsqueeze(0)).squeeze()
            single_feature_dict['exp']=single_exp.detach().cpu().numpy().tolist()
            with open(os.path.join(feature_save_path,single_patch.replace('.png','.json')),'w') as f:
                json.dump(single_feature_dict,f)
    def extract_single_patch(self,seg_patch_path,save_path):
        slide_name=os.path.dirname(seg_patch_path).split('/')[-2]
        single_patch_path=os.path.join(self.patch_path,slide_name,os.path.basename(seg_patch_path).replace('json','png'))

        single_feature_dict={}
        spec_patch_path=single_patch_path
        single_patch=os.path.basename(seg_patch_path).replace('json','png')
        scist_patch=self.scist_preprocessing(spec_patch_path)
        spec_orig_exp_path=os.path.join(self.orig_exp,svs_name,os.path.basename(seg_patch_path).replace('json','npy'))
        orig_exp=torch.Tensor(np.load(spec_orig_exp_path)).squeeze()

        if self.gpu:
            scist_patch=scist_patch.to('cuda:0')
            orig_exp=orig_exp.to('cuda:0')
        single_exp=self.scist(scist_patch.unsqueeze(0),orig_exp.unsqueeze(0)).squeeze()
        single_feature_dict['exp']=single_exp.detach().cpu().numpy().tolist()
        feature_save_path=os.path.join(save_path,svs_name)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)
        with open(os.path.join(feature_save_path,os.path.basename(seg_patch_path)),'w') as f:
            json.dump(single_feature_dict,f)


    def batch_extract(self):
        for svs in os.listdir(self.patch_path):
            assert svs not in os.listdir(self.orig_exp)
            print(svs, ' is processing...')
            self.extract_single_feature(svs)
            print(svs, ' has done')

    def scist_preprocessing(self,patch_path):
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        scist_process = transforms.Compose(
            [transforms.Resize((224, 224), antialias=True),  # resize to 256x256 square
             transforms.ConvertImageDtype(torch.float),
             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # 归一化
             ]
        )
        patch=Image.open(patch_path)
        patch = torch.Tensor(np.array(patch) / 255)
        patch=patch.permute(2,0,1)
        return scist_process(patch)





if __name__ == '__main__':
    patch_path='./01-data/02-patch/'
    coord_path='./01-data/03-coord-added/'
    orig_path='./01-data/05-orig-exp/'
    save_path='./01-data/06-features/'
    extractor=Extract(patch_path,coord_path,orig_path,save_path)

    #1. 从文件夹提取
    # index=0
    # for svs_name in os.listdir('./01-data/04-seg-result/'):
    #     if svs_name not in os.listdir('./01-data/06-features/'):
    #         index+=1
    #         extractor.extract_single_feature(svs_name)
    #         print(f'{index} : {svs_name}')

    # 2. 单独提取一张svs的特征
    # svs_name='TCGA-DD-AAVR-01Z-00-DX1'
    # extractor.extract_single_feature(svs_name)

    index=0
    seg_path='./01-data/04-seg-result/'
    test_save_path='./01-data/test/'
    for svs_name in os.listdir(seg_path):
        existed_path=os.path.join(save_path,svs_name)
        existed_patch_lst=os.listdir(existed_path)
        for seg_output in os.listdir(os.path.join(seg_path,svs_name,'json')):
            if seg_output not in existed_patch_lst:
                extractor.extract_single_patch(os.path.join(seg_path,svs_name,'json',seg_output),test_save_path)
