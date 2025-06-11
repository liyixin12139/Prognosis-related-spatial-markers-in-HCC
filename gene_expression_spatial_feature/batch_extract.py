import os
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
sys.path.append('/2data/liyixin/03-PPS与空间基因相关性分析/code/')
import pandas as pd
from PIL import Image
import anndata as ad
import re
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde, kurtosis, skew
from gene_expression_spatial_feature._function import *
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    '''
    假设已经对每张slide提取了scist的exp特征
    :return:
    '''
    # gene_list_path = '/2data/liyixin/HE2ST/04Results/02-单细胞参考预测基因表达/01-data/04-Liver/liver_hvg_cut_200_minus3.npy'
    # gene_list = list(np.load(gene_list_path))
    #只计算26个高预测性基因的特征
    sig_gene_path='/2data/liyixin/03-PPS与空间基因相关性分析/03-results/01-352个slides上通过p值计算出的高预测性基因/01-raw-pValue-005.npy'
    sig_gene_lst=np.load(sig_gene_path,allow_pickle=True)
    with open('/2data/liyixin/03-PPS与空间基因相关性分析/03-results/02-基因高低表达分类阈值/05-GMM-autoselect-thresholds.json','r') as f:
        sig_gene_threshold_dict=json.load(f)  #提取出所有特征后，要对所有基因找阈值(中位数)
    feature_fold='/2data/liyixin/03-PPS与空间基因相关性分析/01-data/06-features/'
    coord_fold='/2data/liyixin/03-PPS与空间基因相关性分析/01-data/03-coord-added/'
    binary_img_save_path='/2data/liyixin/03-PPS与空间基因相关性分析/03-results/13-钟老师二次打回来后重新做-GMM选择的二分类阈值-binary-img/'
    global_local_save_path='/2data/liyixin/03-PPS与空间基因相关性分析/03-results/14-GMM-提取的空间特征/'
    idx=0
    for single_slide in os.listdir(feature_fold):
        if single_slide not in os.listdir(binary_img_save_path):
        # if single_slide=='TCGA-G3-A6UC-01Z-00-DX1':
            print(f'{single_slide} is processing...')
            binary_slide_fold=os.path.join(binary_img_save_path,single_slide)
            mkdir(binary_slide_fold)
            exp_path=os.path.join(feature_fold,single_slide)
            coord_path=os.path.join(coord_fold,single_slide+'_patch_centers.csv')
            adata=generate_adata(exp_path,coord_path)
            for single_gene in sig_gene_lst: #只提取26个基因的全局和局部特征
                # if single_gene=='CD19':
                print(single_gene)
                global_fea_dict=global_feature(adata,single_gene,sig_gene_threshold_dict[single_gene]) #5个特征
                save_binary_img(adata,single_gene,os.path.join(binary_slide_fold,single_gene+'.png'))
                local_feature_dict=local_feature(binary_slide_fold,single_gene)  #7个特征
                global_fea_dict.update(local_feature_dict)
                global_local_spaital_savepath_for_gene=os.path.join(global_local_save_path,single_slide)
                mkdir(global_local_spaital_savepath_for_gene)
                with open(os.path.join(global_local_spaital_savepath_for_gene,single_gene+'.json'),'w') as f:
                    json.dump(global_fea_dict,f)
            idx+=1
            print(idx,f' {single_slide} has done!')
    #最终对每一个slide的每一个基因都保存为一个json文件，包括12个特征
if __name__ == '__main__':
    main()