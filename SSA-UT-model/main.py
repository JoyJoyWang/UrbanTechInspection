import os
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from pipeline import semantic_annotation_pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np
from PIL import ImageOps
import gc

def load_filename_with_extensions(data_path, filename):
    full_file_path = os.path.join(data_path, filename)
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    for ext in image_extensions:
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', default='data/', help='specify the root path of images and masks')
    parser.add_argument('--out_dir', default='output', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
    parser.add_argument('--save_json', default=True, action='store_true', help='whether to save annotated json')
    parser.add_argument('--sam', default=True, action='store_true',
                        help='use SAM but not given annotation json, default is False')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth',
                        help='specify the root path of SAM checkpoint')
    parser.add_argument('--light_mode', default=True, action='store_true', help='use light mode')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for processing')
    args = parser.parse_args()
    return args


class ImageDataset(Dataset):
    def __init__(self, img_dir, out_dir, max_size=1024):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.max_size = max_size

        # avoid process repeatedly
        processed_files = {os.path.splitext(f)[0] for f in os.listdir(out_dir)}

        self.img_labels = [
            img_name for img_name in os.listdir(img_dir)
            if (img_name.endswith('.png') or img_name.endswith('.jpg')) and
               f"{os.path.splitext(img_name)[0]}_wall_mask.png" not in processed_files
        ]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")

        # Preserve orientation
        image = ImageOps.exif_transpose(image)

        # Calculate the aspect ratio
        width, height = image.size
        aspect_ratio = width / height

        # Resize the image while maintaining the aspect ratio
        if width > height:
            new_width = min(self.max_size, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(self.max_size, height)
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Convert to tensor
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image, self.img_labels[idx].split('.')[0]




def main(rank, args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    torch.cuda.empty_cache()
    scaler = GradScaler()
    if args.light_mode:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    else:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    if args.light_mode:
        oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_tiny").to(device)
    else:
        oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large").to(device)

    oneformer_coco_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(
        device)

    # if args.light_mode:
    #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    # else:
    #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    if args.light_mode:
        clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd16")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16").to(device)
        clipseg_processor.image_processor.do_resize = False
    else:
        clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
        clipseg_processor.image_processor.do_resize = False

    if args.sam:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(device)
        if args.light_mode:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
                output_mode='coco_rle'
            )
        else:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
                output_mode='coco_rle',
            )
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        filenames = [fn_.replace('.' + fn_.split('.')[-1], '') for fn_ in os.listdir(args.data_dir) if
                     '.' + fn_.split('.')[-1] in image_extensions]
    else:
        mask_generator = None
        filenames = [fn_[:-5] for fn_ in os.listdir(args.data_dir) if '.json' in fn_]

    if rank == 0:
        print('Total number of files: ', len(filenames))


    dataset = ImageDataset(args.data_dir, args.out_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for batch in dataloader:
        imgs,filename = batch
        img=imgs[0]
        filename=filename[0]

        with torch.no_grad():
            semantic_annotation_pipeline(img,filename, args.data_dir, args.out_dir, device, save_img=args.save_img, save_json=args.save_json,
                                         clip_processor=clip_processor, clip_model=clip_model,
                                         oneformer_ade20k_processor=oneformer_ade20k_processor,
                                         oneformer_ade20k_model=oneformer_ade20k_model,
                                         oneformer_coco_processor=oneformer_coco_processor,
                                         oneformer_coco_model=oneformer_coco_model,

                                         clipseg_processor=clipseg_processor, clipseg_model=clipseg_model,
                                         mask_generator=mask_generator)
        gc.collect()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # try:
    main(0, args)
    # except Exception as e:
    #     print(f"Error encountered: {e}. ")



