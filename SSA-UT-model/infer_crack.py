import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16,input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils_crack import load_unet_vgg16
from tqdm import tqdm
import shutil
import cv2
import matplotlib.colors as mcolors


def evaluate_img(model, img,img_name):

    # # Create a white background image
    # white_bg = np.ones_like(img, dtype=np.uint8) * 255
    #
    # # Apply the inverted binary mask to your white background image
    # masked_image = cv2.bitwise_and(white_bg, white_bg, mask=binary_mask)
    #
    # # Add the original image where the mask is non-zero
    # masked_image += cv2.bitwise_and(img, img, mask=~binary_mask)

    input_width, input_height = input_size[0], input_size[1]
    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img,img_name):
    wall_mask_path = os.path.join("./output", img_name + "_wall_mask.png")
    wall_mask_image = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)

    if wall_mask_image is None:
        print("Error: Failed to read wall mask image:", wall_mask_path)
        return None  # Return early if unable to read the mask image

    # Resize the wall mask image to match the dimensions of the input image
    wall_mask_image = cv2.resize(wall_mask_image, (img.shape[1], img.shape[0]))
    # Convert the wall mask image to a binary mask
    kernel = np.ones((20, 20), np.uint8)
    _, binary_mask = cv2.threshold(wall_mask_image, 127, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.bitwise_not(binary_mask)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    binary_mask = cv2.bitwise_not(binary_mask)

    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    # if img_width < input_width or img_height < input_height:
    #     return evaluate_img(model, img)

    stride_ratio = 1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()
        coords = patch_locs[i]
        binary_mask_patch=binary_mask[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width]
# [N, 1, H, W]
        masks_pred = model(X)
        black_indices = (binary_mask_patch == 0)
        masks_pred=masks_pred[0,0]
        masks_pred[black_indices] = float('-inf')
        mask = F.sigmoid(masks_pred).data.cpu().numpy()
        # preds.append(masks_pred.data.cpu().numpy())
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        # probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response[0][0]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
    # black_indices = (binary_mask == 0)


    # min_value = probability_map.min()
    # mean=np.mean(probability_map)
    # std=np.std(probability_map)
    # probability_map_standardized = (probability_map-mean)/std
    # min_value=np.min(probability_map_standardized)
    # probability_map[black_indices] = float('-inf')
    # probability_map_sigmoid = F.sigmoid(torch.tensor(probability_map+1))
    plt.imshow(probability_map, cmap='gray')  # Use cmap='gray' if the image is grayscale
    plt.title('Image Title')  # Set a title for the image
    plt.colorbar()  # Add a colorbar to show the intensity scale
    plt.show()

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

def save_heatmap(probability_map, filename, colormap='viridis'):
    height, width = probability_map.shape
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(probability_map, cmap=colormap)
    ax.axis('off')
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_overlay(img, mask, filename, alpha=0.4):
    height, width = img.shape[0], img.shape[1]
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(img)
    ax.imshow(mask, cmap=transparent_bwr_cmap, alpha=alpha)
    ax.axis('off')
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# 创建一个自定义的 colormap

def transparent_cmap(cmap, alpha_zero=0):
    ncolors = cmap.N
    color_array = cmap(np.arange(ncolors))

    alphas = np.ones(ncolors)
    alphas[0] = alpha_zero
    color_array[:, -1] = alphas

    map_object = mcolors.ListedColormap(color_array)
    return map_object


bwr_cmap = plt.cm.jet
transparent_bwr_cmap = transparent_cmap(bwr_cmap)

def draw_boxes(img, mask, threshold=0.1):
    contours, _ = cv.findContours((mask > threshold).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_with_boxes = img.copy()
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red boxes with line width=2
    img_with_boxes_rgb = cv.cvtColor(img_with_boxes, cv.COLOR_BGR2RGB)

    return img_with_boxes_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, default='./data',help='input dataset directory')
    parser.add_argument('-model_path', type=str, default='./ckp/model_unet_vgg_16_best.pt ',help='trained model path')
    parser.add_argument('-model_type', type=str, choices=['vgg16'],default='vgg16')
    parser.add_argument('-out_viz_dir', type=str, default='./output/test_results', required=False,help='visualization output dir')
    parser.add_argument('-out_pred_dir', type=str, default='./output/test_classification', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.02 , help='threshold to cut off crack response')

    args = parser.parse_args()

    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)

    out_crack_dir = os.path.join(args.out_pred_dir, 'with_crack')
    os.makedirs(out_crack_dir, exist_ok=True)

    out_without_crack_dir = os.path.join(args.out_pred_dir, 'without_crack')
    os.makedirs(out_without_crack_dir,exist_ok=True)

    out_wall_masks_dir = os.path.join(args.out_pred_dir, 'wall_masks')
    os.makedirs(out_wall_masks_dir ,exist_ok=True)

    out_heatmap_dir = os.path.join(args.out_pred_dir, 'heatmaps')
    os.makedirs(out_heatmap_dir, exist_ok=True)

    out_overlay_dir = os.path.join(args.out_pred_dir, 'overlays')
    os.makedirs(out_overlay_dir, exist_ok=True)

    out_box_dir = os.path.join(args.out_pred_dir, 'box')
    os.makedirs(out_box_dir, exist_ok=True)

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):
        #print(str(path))

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img(model, img_0,path.name.split('.')[0])


        if args.out_pred_dir != '' and args.out_viz_dir != '':
            ## save masks of compressed full image
            cv.imwrite(filename=join(out_wall_masks_dir, f'{path.stem}_full.jpg'), img=(prob_map_full * 255).astype(np.uint8))

            img_1 = img_0
            prob_map_patch = evaluate_img_patch(model, img_1,path.name.split('.')[0])
            prob_map_patch=np.array(prob_map_patch)

            ##save patch masks
            cv.imwrite(filename=join(out_wall_masks_dir, f'{path.stem}_patch.jpg'), img=(prob_map_patch * 255).astype(np.uint8))

            prob_map_viz_patch = prob_map_patch.copy()
            prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
            prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0

            fig = plt.figure(figsize=(8, 5))
            plt.axis('off')
            st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="x-large")
            ax = fig.add_subplot(131)
            ax.axis('off')
            ax.imshow(img_1)
            ax = fig.add_subplot(132)
            ax.axis('off')
            ax.imshow(prob_map_viz_patch,cmap='Reds')
            ax = fig.add_subplot(133)
            ax.imshow(img_1)
            ax.axis('off')
            ax.imshow(prob_map_viz_patch, cmap=transparent_bwr_cmap,alpha=0.9)

            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0

            # ax = fig.add_subplot(234)
            # ax.imshow(img_0)
            # ax.axis('off')
            # ax = fig.add_subplot(235)
            # ax.imshow(prob_map_viz_full)
            # ax.axis('off')
            # ax = fig.add_subplot(236)
            # ax.imshow(img_0)
            # ax.axis('off')
            # ax.imshow(prob_map_viz_full, alpha=0.4)


            ##save comparison images with 6 subplots
            plt.savefig(join(args.out_viz_dir, f'{path.stem}.jpg'), dpi=500)
            plt.close('all')

            ## save heatmap
            save_heatmap(prob_map_viz_patch, join(out_heatmap_dir, f'{path.stem}_heatmap_patch.jpg'), colormap='hot')

            ## save overlay
            save_overlay(img_1, prob_map_viz_patch, join(out_overlay_dir, f'{path.stem}_HEATMAP.jpg'),0.4)

            ## save boxes
            img_with_boxes = draw_boxes(img_1, prob_map_viz_patch)
            cv.imwrite(join(out_box_dir, f'{path.stem}_boxes_patch.jpg'), img_with_boxes)


        # do classification
        crack_detected = np.mean(prob_map_viz_patch > args.threshold) > 0.005
        if crack_detected:
            img_path = str(path)
            target_img_path = join(out_crack_dir, path.name)
            shutil.copy(img_path, target_img_path)
        else:
            img_path = str(path)
            target_img_path = join(out_without_crack_dir, path.name)
            shutil.copy(img_path, target_img_path)

        gc.collect()