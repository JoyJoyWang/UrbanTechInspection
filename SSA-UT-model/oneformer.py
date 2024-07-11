import torch
import torch.nn.functional as F


def post_process_semantic_segmentation_custom(
        outputs, target_sizes = None,k=3
) -> "torch.Tensor":
    """
    Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
    PyTorch.

    Args:
        outputs ([`MaskFormerForInstanceSegmentation`]):
            Raw outputs of the model.
        target_sizes (`List[Tuple[int, int]]`, *optional*):
            List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
            final size (height, width) of each prediction. If left to None, predictions will not be resized.
    Returns:
        `List[torch.Tensor]`:
            A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
            corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
            `torch.Tensor` correspond to a semantic class id.
    """
    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    # Resize logits and compute semantic segmentation maps
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        semantic_segmentation = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            topk_probs,topk_classes = torch.topk(resized_logits[0, :, :], k=k, dim=0)
            # semantic_map = resized_logits[0].argmax(dim=0)
            # semantic_segmentation.append(semantic_map)
            semantic_segmentation.append(topk_classes)
    else:
        semantic_segmentation = segmentation.argmax(dim=1)
        semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

    return semantic_segmentation

def oneformer_coco_segmentation(image, oneformer_coco_processor, oneformer_coco_model, rank):
    inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_maps = post_process_semantic_segmentation_custom(
        outputs, target_sizes=[image.size[::-1]])
    predicted_semantic_map=predicted_semantic_maps[0]
    return predicted_semantic_map

def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = post_process_semantic_segmentation_custom(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(image, oneformer_cityscapes_processor, oneformer_cityscapes_model, rank):
    inputs = oneformer_cityscapes_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = oneformer_cityscapes_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map