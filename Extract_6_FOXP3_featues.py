# import torch
# print(torch.cuda.is_available())
import os
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from PIL import Image
import anndata as ad
import re

def main(slide_name):
    #整理表达的数据，到adata里
    exp_path=os.path.join('./01-data/06-features/',slide_name)
    coord_path=os.path.join('./01-data/03-coord-added/',slide_name+'_patch_centers.csv')
    def extract_number(filename):
        # 使用正则表达式提取数字
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    exp_files=os.listdir(exp_path)
    exp=sorted(exp_files,key=extract_number)
    num_patch=len(exp)
    exp_array=np.zeros((num_patch,183))
    for i,file in enumerate(exp):
        with open(os.path.join(exp_path,file).replace('npy','json'),'r') as f:
            single_exp=json.load(f)
        exp_array[i]=single_exp['exp']
    adata=ad.AnnData(exp_array)
    coord=pd.read_csv(coord_path)
    coord.iloc[:,0]=[i.split('/')[-1] for i in coord.iloc[:,0]]
    coord.index=coord.iloc[:,0]
    spatial_array=np.zeros((num_patch,2))
    selected_patch=[i.replace('json','png') for i in exp]
    selected_coord=coord.loc[selected_patch]
    spatial=np.concatenate((np.array(selected_coord.iloc[:,1]).reshape(num_patch,1),np.array(selected_coord.iloc[:,2]).reshape(num_patch,1)),axis=1)
    adata.obsm['spatial']=spatial
    gene_path='./01-data/04-Liver/liver_hvg_cut_200_minus3.npy'
    gene=list(np.load(gene_path))
    adata.var_names=gene
    adata.obs.index = adata.obs.index.astype(int)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    def normalize(gene_name):
        tumor_score = adata[:, gene_name].X
        # Min-Max
        tumor_score_normalized = (tumor_score - tumor_score.min()) / (tumor_score.max() - tumor_score.min() + 1e-5)
        return tumor_score_normalized
    genes=['SERPINB3','BTF3','GSTA1','GPC3','ACTA2','CD34','PDGFRB','PDPN','HPX']
    for gene in genes:
        normalize_value=normalize(gene)
        new_name=gene+'_normalized'
        adata.obs[new_name]=normalize_value
    adata.obs['tumor_score']=(adata.obs['SERPINB3_normalized']+adata.obs['HPX_normalized']+adata.obs['GSTA1_normalized'])/3
    adata.obs['stromal']=(adata.obs['ACTA2_normalized']+adata.obs['PDGFRB_normalized']+adata.obs['CD34_normalized']+adata.obs['PDPN_normalized'])/4
    adata.obs['stromal_tissue']=adata.obs['stromal']/adata.obs['tumor_score']
    adata.obs['final'] = (
                (adata.obs['tumor_score'] * adata.obs['tumor_score']) / (adata.obs['stromal'] * adata.obs['stromal']))
    from sklearn.neighbors import NearestNeighbors

    stromal_val = adata.obs['stromal_tissue'].values
    final_val = adata.obs['final'].values

    stromal_thresh = np.percentile(stromal_val, 70)
    final_thre = np.percentile(final_val, 50)
    tissue_type = np.full(final_val.shape, 'Other', dtype=object)

    mask_stroma = (stromal_val >= stromal_thresh)
    tissue_type[mask_stroma] = 'Stroma'
    mask_tumor = (final_val > final_thre)
    tissue_type[mask_tumor] = 'Tumor'

    spatial_coords = adata.obsm['spatial']

    n_neighbors=24
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    neighbor_tumor_fraction_threshold = 0.5
    neighbor_stroma_fraction_threshold = 0.3  

    for i in range(len(tissue_type)):
        if tissue_type[i] == 'Tumor':
            neighbor_inds = indices[i, 1:]
            tumor_neighbors = np.sum(tissue_type[neighbor_inds] == 'Tumor')
            stroma_neighbors = np.sum(tissue_type[neighbor_inds] == 'Stroma')
            fraction_tumor = tumor_neighbors / n_neighbors
            fraction_stroma = stroma_neighbors / n_neighbors

            if fraction_tumor < neighbor_tumor_fraction_threshold or fraction_stroma > neighbor_stroma_fraction_threshold:
                tissue_type[i] = 'Other'
    # 将结果保存到 adata.obs 中
    adata.obs['tissue_type'] = tissue_type
    tumor_adata = adata[adata.obs['tissue_type'] == 'Tumor']

    # 2. 提取 FLXP3 的表达值
    flxp3_index = tumor_adata.var_names.get_loc('FOXP3')
    flxp3_expr = tumor_adata.X[:, flxp3_index].toarray().flatten() 

    # 3. 计算 90 分位数阈值
    foxp3_expr = adata[:, 'FOXP3'].X
    if hasattr(foxp3_expr, "toarray"):
        foxp3_expr = foxp3_expr.toarray().flatten()
    else:
        foxp3_expr = np.array(foxp3_expr).flatten()

    # 计算 20% 和 80% 分位数阈值
    threshold_low = np.percentile(foxp3_expr, 10)
    threshold_high = np.percentile(foxp3_expr, 90)

    # 4. 筛选出高表达的 spot 并计算比例
    high_expr_mask_tumor = flxp3_expr > threshold_high
    tumor_high_propotion = np.sum(high_expr_mask_tumor) / tumor_adata.n_obs

    low_expr_mask_tumor = flxp3_expr < threshold_low
    tumor_low_propotion = np.sum(low_expr_mask_tumor) / tumor_adata.n_obs
    tumor_high_low_propotion=np.sum(high_expr_mask_tumor)/np.sum(low_expr_mask_tumor)
    # print('ratio: ',tumor_high_low_propotion)

    stroma_adata = adata[adata.obs['tissue_type'] == 'Stroma']

    # 2. 提取 FLXP3 的表达值
    flxp3_index = stroma_adata.var_names.get_loc('FOXP3')
    flxp3_expr = stroma_adata.X[:, flxp3_index].toarray().flatten()  


    # 4. 筛选出高表达的 spot 并计算比例
    high_expr_mask_stroma = flxp3_expr > threshold_high
    stroma_high_ratio = np.sum(high_expr_mask_stroma) / stroma_adata.n_obs

    low_expr_mask_stroma = flxp3_expr < threshold_low
    stroma_low_ratio = np.sum(low_expr_mask_stroma) / stroma_adata.n_obs
    stroma_high_low_ratio=np.sum(high_expr_mask_stroma)/np.sum(low_expr_mask_stroma)

    tumor_ratio=tumor_adata.n_obs/adata.n_obs
    stroma_ratio=stroma_adata.n_obs/adata.n_obs
    tumor_high_ratio=tumor_high_propotion
    tumor_low_ratio=tumor_low_propotion
    tumor_high_low=tumor_high_low_propotion
    stroma_high_ratio=stroma_high_ratio
    stroma_low_ratio=stroma_low_ratio
    stroma_high_low=stroma_high_low_ratio
    tumor_stroma_high=np.sum(high_expr_mask_tumor)/np.sum(high_expr_mask_stroma)
    final_ratio=tumor_high_low/stroma_high_low
    return {
        'slide_name':slide_name,
        'tumor_ratio':tumor_ratio,
        'stroma_ratio':stroma_ratio,
        'tumor_high':tumor_high_ratio,
        'tumor_low':tumor_low_ratio,
        'tumor_high_low':tumor_high_low,
        'stroma_high':stroma_high_ratio,
        'stroma_low':stroma_low_ratio,
        'stroma_high_low':stroma_high_low,
        'tumor_stroma_high':tumor_stroma_high,
        'final_ratio':final_ratio
    }

if __name__ == '__main__':
    df=pd.read_csv('./01-data/10-clinical-info/01--clinical-info.csv',index_col=0)
    data=[]
    i=0
    for slide in df.index:
        fea=main(slide)
        data.append(fea)
        i+=1
        print(i,slide)
    df_fea=pd.DataFrame(data,columns=list(fea.keys()))
    df_fea.index=df_fea.iloc[:,0]
    df_merge=pd.concat([df,df_fea],axis=1)
    df_merge.to_csv('./feature-neighbor-24.csv')