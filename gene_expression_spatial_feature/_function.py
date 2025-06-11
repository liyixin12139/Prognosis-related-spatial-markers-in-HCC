import os
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from PIL import Image
import anndata as ad
import re
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde, kurtosis, skew
custom_palette = {
    'high': '#FF5733',  # 红色
    'b': '#33FF57',  # 绿色
    'low': '#3357FF'   # 蓝色
}
def extract_number(filename):
    # 使用正则表达式提取数字
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0
def generate_adata(exp_path,coord_path):
    #整理表达的数据，到adata里
    # exp_path='/2data/liyixin/03-PPS与空间基因相关性分析/01-data/06-features/TCGA-2Y-A9H1-01Z-00-DX1/'
    # coord_path='/2data/liyixin/03-PPS与空间基因相关性分析/01-data/03-coord/TCGA-2Y-A9H1-01Z-00-DX1_patch_centers.csv'
    exp_files=os.listdir(exp_path)
    exp=sorted(exp_files,key=extract_number)
    num_patch=len(exp)
    exp_array=np.zeros((num_patch,183))
    for i,file in enumerate(exp):
        with open(os.path.join(exp_path,file).replace('npy','json'),'r') as f:
            single_exp=json.load(f)
        exp_array[i]=single_exp['exp']
    #加上spatial数据
    adata=ad.AnnData(exp_array)
    coord=pd.read_csv(coord_path)
    coord.iloc[:,0]=[i.split('/')[-1] for i in coord.iloc[:,0]]
    coord.index=coord.iloc[:,0]
    selected_patch=[i.replace('json','png') for i in exp]
    selected_coord=coord.loc[selected_patch]
    spatial=np.concatenate((np.array(selected_coord.iloc[:,1]).reshape(num_patch,1),np.array(selected_coord.iloc[:,2]).reshape(num_patch,1)),axis=1)
    adata.obsm['spatial']=spatial
    #加上基因名
    gene_path='/2data/liyixin/HE2ST/04Results/02-单细胞参考预测基因表达/01-data/04-Liver/liver_hvg_cut_200_minus3.npy'
    gene_list=list(np.load(gene_path))
    adata.var_names=gene_list
    return adata

def global_feature(adata,gene,threshold):
    #1. 全局特征
    #对某一个特定基因来说，例如CD27
    # gene='CD27'
    feature_dict={}
    expression_values = adata.obs_vector(gene)
    # print(expression_values.shape)
    expressions = np.array(expression_values)  # 基因表达值
    global_mean = np.mean(expressions)  #某个基因的全局平均表达值
    global_std = np.std(expressions) #某个基因的全局std
    # Skewness: 偏度
    kde_skewness = skew(expressions)
    # Kurtosis: 峰度
    kde_kurtosis = kurtosis(expressions, fisher=False)  # 使用原始峰度（非 Fisher 形式）

    # 加一个高表达比例 这里的高低表达阈值需要在352张slide上的所有spots找中位数，或者上下1/4位数，一定是全局的，而不是单张的
    # threshold=0.395 #举个例子
    adata.obs[f'{gene}_high_low'] = np.where(expression_values >= threshold, 'high',
                                             np.where(expression_values < threshold, 'low', 'medium'))  # 某个基因的高表达比例
    high_ratio = (list(adata.obs[f'{gene}_high_low']).count('high')) / len(list(adata.obs[f'{gene}_high_low']))
    # print('1. global_mean: ',global_mean)
    # print('2. global_std: ',global_std)
    # print('3. high_ratio: ',high_ratio)
    feature_dict['global_mean']=global_mean
    feature_dict['global_std']=global_std
    feature_dict['skewness']=kde_skewness
    feature_dict['kurtosis']=kde_kurtosis
    feature_dict['high_ratio']=high_ratio

    return feature_dict


def save_binary_img(adata,gene,save_path):
    #save_path应该是到slide文件夹下
    # gene = "CD27"  # 目标基因
    # fig,ax=plt.subplots(figsize=(10,6))
    # 定义自定义颜色
    # ax=sc.pl.spatial(adata, color=f'{gene}_high_low', legend_loc='on data',spot_size=200,palette=custom_palette)
    ax=sc.pl.spatial(adata, color=f'{gene}_high_low', spot_size=200,palette=custom_palette,show=False,return_fig=True,title=None,legend_loc=None)
    ax=ax[0]
    # 去除坐标轴和边框
    ax.axis('off')  # 去除坐标轴
    for spine in ax.spines.values():  # 去除边框
        spine.set_visible(False)
    ax.set_title("")
    # # 保存只包含点的图
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # # 关闭图形以释放内存
    plt.close()

def local_feature(slide_path,gene):
    # 2. 开始提取
    # 局部总共提取7个特征
    # 边界总长度
    feature_dict={}
    from scipy.signal import convolve2d
    img_path=os.path.join(slide_path,gene+'.png')
    img = Image.open(img_path)
    img = np.array(img)
    binary_array = (img[:, :, 0] > 200).astype(float)
    # binary_array = np.stack([binary_array] * 2, axis=-1)
    # print(binary_array.shape)
    kernel = np.ones((25, 25))
    # convolved_image = convolve2d(binary_array, kernel, mode='same', boundary='fill', fillvalue=0)
    convolved_image = convolve2d(binary_array, kernel, mode='valid', boundary='fill', fillvalue=0)
    output = np.where((convolved_image >= 300) & (convolved_image <= 320), 1, 0)
    boundary_length = np.sum(output)
    boundary_ratio = boundary_length / (output.shape[0] * output.shape[1])
    # print(f"1. 边界总长度比例: {boundary_ratio}")
    # entropy
    from scipy.stats import entropy
    hist, _ = np.histogram(output, bins=256, range=(0, 255), density=True)
    # 计算全局熵
    image_entropy = entropy(hist, base=2)
    print('2. entropy: ', image_entropy)

    from scipy.stats import gaussian_kde
    if np.sum(output)<=100: #初步定为1000是为了筛选边界不明显的图
        feature_dict['boundaty_ratio'] = boundary_ratio
        feature_dict['entropy'] = image_entropy
        feature_dict['kde_max'] = 0
        feature_dict['kde_min'] = 0
        feature_dict['kde_entropy'] = 0
        feature_dict['kde_skewness'] = 0
        feature_dict['kde_kurtosis'] = 0
        return feature_dict
    else:
        coords = np.argwhere(output)
        # 核密度估计
        # print(output)
        # print(np.sum(output))
        # print(coords.T)
        kde = gaussian_kde(coords.T)
        density = kde(coords.T)

        max_val = np.max(density)

        # Min: KDE 密度分布的最小值
        min_val = np.min(density)

        # Entropy: 基于 KDE 密度的归一化分布计算熵
        density_normalized = density / np.sum(density)  # 确保密度归一化
        kde_entropy = entropy(density_normalized, base=2)

        # Skewness: 偏度
        kde_skewness = skew(density)

        # Kurtosis: 峰度
        kde_kurtosis = kurtosis(density, fisher=False)  # 使用原始峰度（非 Fisher 形式）
        # print(f"3. Max: {max_val}")
        # print(f"4. Min: {min_val}")
        # print(f"5. Entropy: {kde_entropy}")
        # print(f"6. Skewness: {kde_skewness}")
        # print(f"7. Kurtosis: {kde_kurtosis}")
        # 输出结果
        feature_dict['boundaty_ratio']=boundary_ratio
        feature_dict['entropy'] = image_entropy
        feature_dict['kde_max'] = max_val
        feature_dict['kde_min'] = min_val
        feature_dict['kde_entropy'] = kde_entropy
        feature_dict['kde_skewness'] = kde_skewness
        feature_dict['kde_kurtosis'] = kde_kurtosis
        return feature_dict



