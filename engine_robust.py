import math, sys, time, json, os, tempfile, shutil, glob, torch, numpy as np
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor): return obj.cpu().tolist()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

@torch.inference_mode()
def evaluate_robust(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    temp_dir = tempfile.mkdtemp()
    try:
        for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            serializable_res = convert_to_serializable(res)
            with open(os.path.join(temp_dir, f'results_{batch_idx}.json'), 'w') as f:
                json.dump(serializable_res, f)
            metric_logger.update(model_time=model_time)
        print("\nMenggabungkan hasil prediksi dari disk...")
        all_results = {}
        for f_path in glob.glob(os.path.join(temp_dir, '*.json')):
            with open(f_path, 'r') as f:
                data = json.load(f)
                all_results.update({int(k): v for k, v in data.items()})
        final_res_tensors = {}
        for img_id, preds in all_results.items():
            final_res_tensors[img_id] = {
                'boxes': torch.tensor(preds['boxes']),
                'scores': torch.tensor(preds['scores']),
                'labels': torch.tensor(preds['labels'], dtype=torch.int64)
            }
        if final_res_tensors:
            coco_evaluator.update(final_res_tensors)
    finally:
        shutil.rmtree(temp_dir)
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator