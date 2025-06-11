import os
import json
import scprep as scp
import numpy as np
import pandas as pd
from data_preprocess._03_batch_segment import mkdir


def count_celltype(json_path,id2tissue_dict)->dict:

    with open(json_path,'r') as f:
        json_file = json.load(f)
    json_nuc_dict = json_file['nuc']

    count_dict = {}
    for celltype in list(id2tissue_dict.values()):
        count_dict[celltype] = 0

    for cell_i in json_nuc_dict.values():
        sin_celltype = id2tissue_dict[str(cell_i['type'])]
        count_dict[sin_celltype]+=1
    f.close()
    return count_dict


def substitute_gene(gene_list,orig_gene,update_gene):
    gene_list_update = [update_gene if i==orig_gene else i for i in gene_list]
    return gene_list_update

def process_ref(ref_path)->pd.DataFrame:
    ref = pd.read_csv(ref_path)
    ref.set_index('Unnamed: 0', inplace=True)
    return ref


def create_orig_exp(count_dict,sincell_ref):
    weight_dict = {}
    her2_ref_celltype = ['neoplastic', 'inflammatory', 'connective','non-neoplastic epithelial'] 
    ref_sum = sum([v for k, v in count_dict.items() if k in her2_ref_celltype])
    orig_exp = np.zeros((1,sincell_ref.shape[1]))
    if ref_sum==0:
        for celltype in her2_ref_celltype:
            orig_exp += np.array(sincell_ref[sincell_ref.index==celltype])*(1/len(her2_ref_celltype))
        return orig_exp
    else:
        for k, v in count_dict.items():
            if k in her2_ref_celltype:
                weight_dict[k] = count_dict[k] / ref_sum
        for k,v in weight_dict.items():
            orig_exp += np.array(sincell_ref[sincell_ref.index==k])*weight_dict[k]
        return orig_exp

def create_orig_exp_for_patch(json_path,predicted_gene_path,ref_path):
    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    result = count_celltype(json_path,id2tissue)
    predicted_genes = list(np.load(predicted_gene_path, allow_pickle=True))
    ref = process_ref(ref_path)
    need_substitute_genes = ['TESMIN','BICDL1','GRK3','ARHGAP45']
    update_genes = ['MTL5','BICD1','ADRBK2','HMHA1']
    for gene_i in range(len(need_substitute_genes)):
        predicted_genes = substitute_gene(predicted_genes,need_substitute_genes[gene_i],update_genes[gene_i])
    ref = ref[predicted_genes] #（cell_num,gene_num）在这里是（5，785）
    orig_exp = create_orig_exp(result,ref)
    return orig_exp

def her_create_orig_exp_for_patch_scp(json_path,predicted_gene_path,ref_path):
    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    result = count_celltype(json_path,id2tissue)

    predicted_genes = list(np.load(predicted_gene_path, allow_pickle=True))

    ref = process_ref(ref_path)

    need_substitute_genes = ['TESMIN','BICDL1','GRK3','ARHGAP45']
    update_genes = ['MTL5','BICD1','ADRBK2','HMHA1']
    for gene_i in range(len(need_substitute_genes)):
        predicted_genes = substitute_gene(predicted_genes,need_substitute_genes[gene_i],update_genes[gene_i])
    ref = ref[predicted_genes] #（cell_num,gene_num）在这里是（5，785）
    ref_scp = np.array(ref.iloc[:3,:])
    ref_scp = scp.transform.log(scp.normalize.library_size_normalize(ref_scp))
    for i in range(ref_scp.shape[0]):
        ref.iloc[i,:] = ref_scp[i]
    orig_exp = create_orig_exp(result,ref)
    return orig_exp

def gen_origExp_for_all_patch(gene_num,svs_name):
    path = os.path.join('./01-data/04-seg-result/',svs_name,'json') 
    ref_path = './01-data/04-Liver/04-single_cell_ref/183_mean_scp.csv'
    save_path = os.path.join('./01-data/05-orig-exp/',svs_name) 
    mkdir(save_path)
    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    ref = process_ref(ref_path)
    for json_file in os.listdir(path):
        spec_path = os.path.join(path,json_file)
        result = count_celltype(spec_path,id2tissue)
        orig_exp = create_orig_exp(result,ref)
        orig_exp = orig_exp.reshape((gene_num,))
        np.save(os.path.join(save_path,json_file.split('.')[0]+'.npy'),orig_exp)





if __name__ == '__main__':

    index=0
    for svs_name in os.listdir('./01-data/04-seg-result/'):
        if svs_name not in os.listdir('./01-data/05-orig-exp/'):
            index+=1
            gen_origExp_for_all_patch(183,svs_name)
            print(f'{index} : {svs_name}')



