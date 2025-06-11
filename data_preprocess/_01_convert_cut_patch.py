import os
import openslide
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
from openslide import OpenSlide
from PIL import Image


def convert_20X(svs_path):
    slide = OpenSlide(svs_path)
    objective_power = float(slide.properties.get("openslide.objective-power"))
    if objective_power == 40.0:
        level_40X = 0
        dimensions_40X = slide.level_dimensions[level_40X]
        dimensions_20X = (dimensions_40X[0] // 2, dimensions_40X[1] // 2)

        region_40X = slide.read_region((0, 0), level_40X, dimensions_40X) 
        region_20X = region_40X.resize(dimensions_20X, Image.Resampling.LANCZOS) 
        return region_20X
    elif objective_power == 20.0:
        level_20X = 0
        dimensions_20X = slide.level_dimensions[level_20X]
        region_20X = slide.read_region((0, 0), level_20X, dimensions_20X)
        return region_20X


def cut_svs(region_20X, patch_save_path, coord_save_path, tcga_name):
    # image = np.array(region_20X)[:, :, :3]
    image=region_20X
    height, width, _ = image.shape
    print(f"图像尺寸: 宽={width}, 高={height}")
    PATCH_SIZE = 224  
    BACKGROUND_THRESHOLD = 0.9
    print("开始切割图像...")
    patch_count = 0
    patch_centers = []

    for y in range(0, height, PATCH_SIZE):
        for x in range(0, width, PATCH_SIZE):
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                continue

            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            _, thresholded = cv2.threshold(gray_patch, 170, 255, cv2.THRESH_BINARY)  # 阈值为230接近白色
            non_background_ratio = 1 - np.sum(thresholded == 255) / (PATCH_SIZE * PATCH_SIZE)

            if non_background_ratio < (1 - BACKGROUND_THRESHOLD): 
                continue

            patch_filename = os.path.join(patch_save_path, f"patch_{patch_count}.png")
            # np.save(patch_filename,patch)
            Image.fromarray(patch).save(patch_filename)

            center_x = x + PATCH_SIZE // 2
            center_y = y + PATCH_SIZE // 2
            patch_centers.append((patch_filename, center_x, center_y))

            patch_count += 1

    print(f"共保存 {patch_count} 个有效 patch。")
    output_csv = os.path.join(coord_save_path, tcga_name + '_' + "patch_centers.csv")
    with open(output_csv, "w") as f:
        f.write("patch_filename,center_x,center_y\n")
        for filename, cx, cy in patch_centers:
            f.write(f"{filename},{cx},{cy}\n")


def get_20x_from_40x(svs_path):

    slide = OpenSlide(svs_path)
    objective_power = float(slide.properties.get("openslide.objective-power"))
    if objective_power == 40.0:
        level_40x = 0
        width_40x, height_40x = slide.level_dimensions[level_40x]
        print(f"40x图像尺寸: 宽 = {width_40x}, 高 = {height_40x}")

        width_20x, height_20x = width_40x // 2, height_40x // 2
        print(f"20x目标图像尺寸: 宽 = {width_20x}, 高 = {height_20x}")

        img_40x = slide.read_region((0, 0), level_40x, (width_40x, height_40x))
        img_40x = img_40x.convert("RGB") 
        img_20x = img_40x.resize((width_20x, height_20x), Image.Resampling.LANCZOS)
        img_matrix = np.array(img_20x)
        print("已成功生成20x的图像数据矩阵。")
        return img_matrix
    elif objective_power == 20.0:
        width_20x, height_20x = slide.level_dimensions[0]
        img_40x = slide.read_region((0, 0), 0, (width_20x, height_20x))
        img_40x = img_40x.convert("RGB")
        img_matrix=np.array(img_40x)
        return img_matrix


def decrease_threshold_cut_patch(img_matrix,orig_coord_path,coord_save_path,patch_save_path,tcga_name):
    height, width, _ = img_matrix.shape
    print(f"图像尺寸: 宽={width}, 高={height}")
    PATCH_SIZE = 224 
    BACKGROUND_THRESHOLD = 0.6
    print("开始切割图像...")

    orig_coord = pd.read_csv(orig_coord_path)
    patch_count = len(orig_coord)
    patch_centers = [(orig_coord.iloc[i, :]) for i in range(len(orig_coord))]
    for y in range(0, height, PATCH_SIZE):
        for x in range(0, width, PATCH_SIZE):
            patch = img_matrix[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                continue
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            threshold = cv2.inRange(gray_patch, 65, 200)
            non_background_ratio = 1 - np.sum(threshold == 0) / (PATCH_SIZE * PATCH_SIZE)
            if non_background_ratio < BACKGROUND_THRESHOLD: 
                continue

            patch_filename = os.path.join(patch_save_path, f"patch_{patch_count}.png")

            center_x = x + PATCH_SIZE // 2
            center_y = y + PATCH_SIZE // 2

            exists = ((orig_coord['center_x'] == center_x) & (orig_coord['center_y'] == center_y)).any()
            if not exists:
                patch_centers.append((patch_filename, center_x, center_y))

                # np.save(patch_filename,patch)
                Image.fromarray(patch).save(patch_filename)

                patch_count += 1
    print(f"初始patch数量: {len(orig_coord)}.")
    print(f"共增加保存 {patch_count - len(orig_coord)} 个有效 patch。")

    output_csv = os.path.join(coord_save_path, tcga_name + '_' + "patch_centers.csv")
    with open(output_csv, "w") as f:
        f.write("patch_filename,center_x,center_y\n")
        for filename, cx, cy in patch_centers:
            f.write(f"{filename},{cx},{cy}\n")

def batch_process(tcga_path, patch_path, coord_path,coord_save_path,high_resolution_slide,not_enough_slide):
    for fold in os.listdir(tcga_path):
        if os.path.isdir(os.path.join(tcga_path, fold)):
            for file in os.listdir(os.path.join(tcga_path, fold)):
                if file.startswith('TCGA') and file[:23] not in high_resolution_slide and file[:23] not in [i.split('_')[0] for i in os.listdir(coord_save_path)] and file[:23] in not_enough_slide:
                    print(fold +'/'+file[:23], ' is processing...')
                    svs_path = os.path.join(tcga_path, fold, file)
                    img_matrix = get_20x_from_40x(svs_path)
                    patch_save_path = os.path.join(patch_path, file[:23])
                    if not os.path.exists(patch_save_path):
                        os.makedirs(patch_save_path)
                    orig_coord_path=os.path.join(coord_path,file[:23]+'_patch_centers.csv')
                    decrease_threshold_cut_patch(img_matrix, orig_coord_path,coord_save_path,patch_save_path, file[:23])



if __name__ == '__main__':
    tcga_path='./01-data/01-TCGA/'
    patch_path='./01-data/02-patch/'
    orig_coord_path='./01-data/03-coord/'
    coord_save_path='./01-data/03-coord-added/'
    batch_process(tcga_path, patch_path, orig_coord_path, coord_save_path,high_slide, not_enough_for_seg_slide)