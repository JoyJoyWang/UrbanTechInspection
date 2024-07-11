import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
import pycocotools.mask as maskUtils
# from tqdm import tqdm
# from collections import OrderedDict
# from prettytable import PrettyTable
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation, oneformer_cityscapes_segmentation
from matplotlib import pyplot as plt
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
# from blip import open_vocabulary_classification_blip
from clip import clip_classification
from clipseg import clipseg_segmentation
from segformer import segformer_segmentation as segformer_func
# from collections import Counter
import cv2

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
    'foggy_driving': oneformer_cityscapes_segmentation
}


# Define alternative functions or libraries for mmcv functionalities
def imread(file_path):
    return Image.open(file_path)

def load(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# def imcrop(img, bbox):
#     img_pil = Image.fromarray(img)
#     return img_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
def imcrop(img, bbox, scale=1.0):
    # Adjust the bbox according to the scale
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    new_width = width * scale
    new_height = height * scale

    # Calculate the new bbox center
    center_x = bbox[0] + width / 2
    center_y = bbox[1] + height / 2

    # Calculate the new bbox coordinates
    new_bbox = [
        center_x - new_width / 2,
        center_y - new_height / 2,
        center_x + new_width / 2,
        center_y + new_height / 2
    ]

    # Ensure the new_bbox coordinates are within the image bounds
    new_bbox = [max(0, coord) for coord in new_bbox]
    new_bbox[2] = min(img.shape[1], new_bbox[2])
    new_bbox[3] = min(img.shape[0], new_bbox[3])

    img_pil = Image.fromarray(img)
    return img_pil.crop((new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]))

def imshow_det_bboxes(img, bboxes=None, labels=None, segms=None, class_names=None, font_size=25, show=True, out_file=None):
    plt.imshow(img)
    plt.axis('off')
    if segms.any():
        unique_labels = np.unique(labels) if labels is not None else []
        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'grey',
                  'orange', 'purple', 'cyan', 'magenta', 'brown',
                  'lightblue', 'lime', 'teal', 'olive']
        color_map = {}
        for i, segm in enumerate(segms):
            # mask = segm
            # color=i % len(colors)
            # plt.contour(mask,cmap='RdYlBu', linewidths=1, alpha=0.5)
            # plt.contourf(mask,cmap='RdYlBu', alpha=0.5)
            if labels is not None and class_names is not None:
                label = labels[i]
                class_name = class_names[label]
                if class_name not in color_map:
                    color_map[class_name] = colors[len(color_map) % len(colors)]
                color = color_map[class_name]
            else:
                color = colors[i % len(colors)]
            colored_mask = np.zeros((*segm.shape, 4))
            colored_mask[segm == 1] = plt.cm.colors.to_rgba(color)
            colored_mask[:, :, 3] = 0.2
            colored_mask[segm == 0] = (0.0, 0.0, 0.0, 0.0)
            plt.imshow(colored_mask)
    if bboxes:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=class_names[labels[i]], color='orange', linewidth=3)
    if class_names:
        plt.legend()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()

def dump(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def decode(mask):
    return maskUtils.decode(mask)



def load_filename_with_extensions(data_path, filename):
    full_file_path = os.path.join(data_path, filename)
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    for ext in image_extensions:
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")

def semantic_annotation_pipeline(img,filename, data_path, output_path, rank, save_img=False, save_json=False,scale_small=1.2,
                                 scale_large=1.2, scale_huge=1.2,
                                 clip_processor=None, clip_model=None, oneformer_ade20k_processor=None,
                                 oneformer_ade20k_model=None, oneformer_coco_processor=None, oneformer_coco_model=None,
                                 clipseg_processor=None, clipseg_model=None,
                                 mask_generator=None):
    print(save_json)
    img_np = img.numpy().transpose(1, 2, 0)
    img = (img_np * 255).astype(np.uint8)
    img_file_path = os.path.join(output_path, f'{filename}.jpg')

    #save the original image with smaller size to do visualization with json file
    cv2.imwrite(img_file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if mask_generator is None:
        anns = load(os.path.join(data_path, filename + '.json'))
    else:
        anns = {'annotations': mask_generator.generate(img)}

    bitmasks, class_names = [], []

    class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(np.array(img)),
                                                                oneformer_coco_processor, oneformer_coco_model, rank)

    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(np.array(img)),
                                                                    oneformer_ade20k_processor,
                                                                    oneformer_ade20k_model, rank)
    exclude_words = {'tank', 'window', 'stairs', 'stairway', 'window-blind',
                     'sand','sky','boat', 'bannister', 'windowpane', 'tower',
                     'fence', 'chair', 'earth','wire','string', 'hose','pipe','tube','handrail','box','rust',"electric wires","ventilation duct","fire escape"}

    wall_building_mask = np.zeros(img.shape[:2], dtype=bool)
    other_masks = []
    # class_ids_from_oneformer_cityscapes = oneformer_cityscapes_segmentation(Image.fromarray(np.array(img)),oneformer_cityscapes_processor,)
    filtered_annotations = []
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        valid_mask_expanded = valid_mask.unsqueeze(0).expand(3, -1, -1)
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask_expanded].view(3, -1)
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask_expanded].view(3, -1)
        # coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        # ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(3).indices
        top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(3).indices
        local_class_names = set()
        local_class_names = set.union(local_class_names,
                                      set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in
                                           top_k_ade20k_propose_classes_ids]))
        local_class_names = set.union(local_class_names, set(([
            CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in
            top_k_coco_propose_classes_ids])))

        # the following is used for open-vocab identification using clip and blip
        patch_small = imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                             scale=scale_small)
        # patch_large = imcrop(img, np.array(
        #     [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
        #                      scale=scale_large)
        patch_huge = imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                            scale=scale_huge)
        valid_mask_huge_crop = imcrop(valid_mask.numpy(), np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                      scale=scale_huge)

        # op_class_list = open_vocabulary_classification_blip(patch_large, blip_processor, blip_model, rank)
        #
        # local_class_list = list(set.union(local_class_names, set(op_class_list)))

        local_class_list=list(local_class_names)+["crack","sky","wall surface","brick-wall","stone-wall","concrete-wall","rope","string","appliance",
                                                  "handrail","metal","handwriting","wire",
                                                  "cable","scrawl","graffiti","pipe","hose","shadow","device",
                                                  "tube","rust","box","electric wires","ventilation duct","fire escape"]
        mask_categories = clip_classification(patch_small, local_class_list,
                                              5 if len(local_class_list) > 5 else len(local_class_list), clip_processor,
                                              clip_model, rank)

        class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model,
                                                    rank).argmax(0)

        valid_mask_huge_crop = torch.tensor(np.array(valid_mask_huge_crop))

        if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
            valid_mask_huge_crop = F.interpolate(
                valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                size=(class_ids_patch_huge.shape[-2], class_ids_patch_huge.shape[-1]),
                mode='nearest').squeeze(0).squeeze(0).bool()

        top_1_patch_huge = torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten()).topk(1).indices
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        ann['class_name'] = str(top_1_mask_category)

        #------------------------

        # ann['class_proposals'] = mask_categories
        ann['class_proposals'] = mask_categories
        # class_names.append(str(top_1_mask_category))

        # if any("wall" in name for name in class_names_list) and not any(exclude_word in name for exclude_word in exclude_words for name in class_names_list):
        #             filtered_annotations.append(ann)
        #             for name in class_names_list:
        #                 if "wall" in name:
        #                     class_names.append(name)
        #                     break
        if ("wall" in ann['class_name'] or "building" in ann['class_name'] or "crack" in ann['class_name']) and ("wall" in mask_categories[0]+mask_categories[1]  or "building" in mask_categories[0]+mask_categories[1] or "crack" in mask_categories[0]+mask_categories[1]):
            if not any(exclude_word in name for exclude_word in exclude_words for name in mask_categories[:1]):
                filtered_annotations.append(ann)
                class_names.append(ann['class_name'])
                wall_building_mask |= valid_mask.numpy()
            else:
                other_masks.append(valid_mask.numpy())
        else:
            other_masks.append(valid_mask.numpy())
        # class_names.append(str(top_1_mask_category))
    # renew the anns
        # Delete variables that are no longer needed
        del coco_propose_classes_ids
        del ade20k_propose_classes_ids
        del top_k_coco_propose_classes_ids
        del top_k_ade20k_propose_classes_ids
        del patch_small
        # del patch_large
        del patch_huge
        del valid_mask_huge_crop
        del mask_categories
        del class_ids_patch_huge

    for other_mask in other_masks:
        wall_building_mask &= ~other_mask

    valid_wall_building_annotations = []
    for ann in filtered_annotations:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        valid_mask &= torch.tensor(wall_building_mask)
        if valid_mask.any() and torch.sum(torch.tensor(valid_mask)) > 100: #min_size_threshold
            ann['segmentation'] = maskUtils.encode(np.asfortranarray(valid_mask.numpy().astype(np.uint8)))
            ann['segmentation']['counts'] = ann['segmentation']['counts'].decode('utf-8')
            valid_wall_building_annotations.append(ann)

    anns['annotations'] = valid_wall_building_annotations

    binary_mask_image = (wall_building_mask * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_path, filename + "_wall_mask.png"), binary_mask_image)
    if save_json:
        dump(anns, os.path.join(output_path, filename + '_semantic.json'))
        print('[Save] save SSA-engine annotation results: ', os.path.join(output_path, filename + '_semantic.json'))
    if save_img:
        for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
        if bitmasks != []:
            imshow_det_bboxes(img,
                                   bboxes=None,
                                   labels=np.arange(len(bitmasks)),
                                   segms=np.stack(bitmasks),
                                   class_names=class_names,
                                   font_size=25,
                                   show=False,
                                   out_file=os.path.join(output_path, filename + '_semantic.png'))

    # Delete variables that are no longer needed
    del img
    del anns
    del class_ids_from_oneformer_coco
    del class_ids_from_oneformer_ade20k

def img_load(data_path, filename, dataset):
    if dataset == 'ade20k':
        img = imread(os.path.join(data_path, filename + '.jpg'))
    elif dataset == 'cityscapes' or dataset == 'foggy_driving':
        img = imread(os.path.join(data_path, filename + '.png'))
    else:
        raise NotImplementedError()
    return img


# def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
#                                         semantic_branch_processor=None, semantic_branch_model=None,
#                                         mask_branch_model=None, dataset=None, id2label=None, model='segformer'):
#     anns = {'annotations': mask_branch_model.generate(img)}
#     h, w, _ = img.shape
#     class_names = []
#
#     if model == 'oneformer':
#         class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
#                                             semantic_branch_model, rank)
#     elif model == 'segformer':
#         class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
#     else:
#         raise NotImplementedError()
#
#     semantc_mask = class_ids.clone()
#
#     anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
#
#     for ann in anns['annotations']:
#         valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
#         propose_classes_ids = class_ids[valid_mask]
#         num_class_proposals = len(torch.unique(propose_classes_ids))
#
#         if num_class_proposals == 1:
#             semantc_mask[valid_mask] = propose_classes_ids[0]
#             ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
#             ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
#             class_names.append(ann['class_name'])
#             continue
#
#         top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
#         top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]
#
#         semantc_mask[valid_mask] = top_1_propose_class_ids
#         ann['class_name'] = top_1_propose_class_names[0]
#         ann['class_proposals'] = top_1_propose_class_names[0]
#         class_names.append(ann['class_name'])
#
#     sematic_class_in_img = torch.unique(semantc_mask)
#     semantic_bitmasks, semantic_class_names = [], []
#
#     anns['semantic_mask'] = {}
#
#     for i in range(len(sematic_class_in_img)):
#         class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
#         class_mask = semantc_mask == sematic_class_in_img[i]
#         class_mask = class_mask.cpu().numpy().astype(np.uint8)
#         semantic_class_names.append(class_name)
#         semantic_bitmasks.append(class_mask)
#         anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(
#             np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
#         anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = \
#         anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
#
#     if save_img:
#         imshow_det_bboxes(img,
#                                bboxes=None,
#                                labels=np.arange(len(sematic_class_in_img)),
#                                segms=np.stack(semantic_bitmasks),
#                                class_names=semantic_class_names,
#                                font_size=25,
#                                show=False,
#                                out_file=os.path.join(output_path, filename + '_semantic.png'))
#         print('[Save] save SSA prediction: ', os.path.join(output_path, filename + '_semantic.png'))
#
#     dump(anns, os.path.join(output_path, filename + '_semantic.json'))


# def eval_pipeline(gt_path, res_path, dataset):
#     logger = None
#     if dataset == 'cityscapes' or dataset == 'foggy_driving':
#         class_names = (
#         'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
#         'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
#     elif dataset == 'ade20k':
#         class_names = (
#         'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet',
#         'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
#         'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk',
#         'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
#         'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
#         'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
#         'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm',
#         'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light',
#         'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt track',
#         'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
#         'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel',
#         'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
#         'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture',
#         'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
#         'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')
#     else:
#         raise NotImplementedError()
#
#     anns_gt = load(gt_path)
#     anns_res = load(res_path)
#
#     if dataset == 'cityscapes':
#         anns_gt = anns_gt['annotations']
#     else:
#         anns_gt = anns_gt['annotations']['objects']
#
#     if dataset == 'cityscapes':
#         anns_res = anns_res['annotations']
#     else:
#         anns_res = anns_res['annotations']['objects']
#
#     x = PrettyTable()
#     x.field_names = ['class_name', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
#
#     eval_res = {}
#     for class_name in class_names:
#         class_id = str([key for (key, value) in CONFIG_ADE20K_ID2LABEL['id2label'].items() if value == class_name][0])
#         eval_res[class_name] = recognize_eval_single_class(anns_gt, anns_res, class_id)
#
#     # per class AP
#     for class_name in class_names:
#         res = eval_res[class_name]
#         x.add_row([class_name, res['AP'], res['AP50'], res['AP75'], res['APs'], res['APm'], res['APl']])
#
#     logger.info('\n' + x.get_string(title="SSA evaluation results"))
#
#     print('\n' + x.get_string(title="SSA evaluation results"))
#
#     # mAP
#     eval_res = {key: value for key, value in eval_res.items() if not key.startswith('AP')}
#     x.add_row(['mAP', sum([res['AP'] for res in eval_res.values()]) / len(eval_res),
#                sum([res['AP50'] for res in eval_res.values()]) / len(eval_res),
#                sum([res['AP75'] for res in eval_res.values()]) / len(eval_res),
#                sum([res['APs'] for res in eval_res.values()]) / len(eval_res),
#                sum([res['APm'] for res in eval_res.values()]) / len(eval_res),
#                sum([res['APl'] for res in eval_res.values()]) / len(eval_res)])
#     logger.info('\n' + x.get_string(title="SSA evaluation results"))
#     print('\n' + x.get_string(title="SSA evaluation results"))
#
#     return eval_res
