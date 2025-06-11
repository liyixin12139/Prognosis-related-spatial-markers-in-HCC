import os
import shutil
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
os.chdir("./hover_net-master/")
patch_path='./01-data/02-patch/'
seg_path='./01-data/04-seg-result/'
tmp_path='./01-data/patch_test/tmp/'
tmp_segResult_path='./01-data/patch_test/tmp_segResult/'

i=0
for fold in os.listdir(patch_path):
    if fold not in os.listdir(tmp_segResult_path):
    # if fold=='TCGA-5C-A9VG-01Z-00-DX1':
        print(fold, ' is processing...')
        mkdir(os.path.join(seg_path, fold,'json'))
        for sin_patch in os.listdir(os.path.join(patch_path,fold)):
            if sin_patch.replace('.png','.json') not in os.listdir(os.path.join(seg_path,fold,'json')):
            # if sin_patch not in os.listdir(os.path.join(orig_patch_path,fold)):
                shutil.copy(os.path.join(patch_path,fold,sin_patch),os.path.join(tmp_path,sin_patch))
        input_dir=os.path.join(tmp_path)
        output_dir=os.path.join(tmp_segResult_path,fold)
        mkdir(output_dir)

        os.system(f"python run_infer.py \
        --gpu='0' \
        --nr_types=6 \
        --type_info_path=./type_info.json \
        --batch_size=128 \
        --model_mode=fast \
        --model_path=./pretrained_model/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --nr_inference_workers=30 \
        --nr_post_proc_workers=30 \
        tile \
        --input_dir={input_dir} \
        --output_dir={output_dir} \
        --mem_usage=0.1 ")

        shutil.rmtree(os.path.join(output_dir,'mat'))
        shutil.rmtree(os.path.join(output_dir,'overlay'))
        i+=1
        print(i, fold, ' is done')
        shutil.rmtree(tmp_path)
        mkdir(tmp_path)


