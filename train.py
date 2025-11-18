import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import sys, json, torch, numpy as np, pandas as pd, gc, argparse, cv2
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import torch.utils.data
sys.path.insert(0, os.path.abspath('.'))
import utils
from engine_robust import train_one_epoch, evaluate_robust as evaluate

# --- PERUBAHAN DIMULAI DI SINI: Integrasi Albumentations ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            # Tambahkan rotasi. Sangat penting untuk mengatasi FN.
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
            
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            
            # CoarseDropout memaksa model melihat bagian roti yang berbeda
            A.CoarseDropout(max_holes=8, max_height=25, max_width=25, p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.2
        ))
    else:
        # Validasi tetap sama
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

class BreadDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(BreadDataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target_list = super(BreadDataset, self).__getitem__(idx)
        
        # Konversi PIL Image ke NumPy array (wajib untuk Albumentations)
        image = np.array(img)
        
        boxes, labels, areas = [], [], []
        
        for ann in target_list:
            x, y, w, h = ann['bbox']
            # Filter anotasi yang tidak valid dari awal
            if w > 0 and h > 0:
                boxes.append([x, y, w, h])
                labels.append(ann['category_id'])
                areas.append(ann['area'])
        
        # Siapkan dictionary untuk augmentasi
        transform_input = {
            'image': image,
            'bboxes': boxes,
            'category_ids': labels
        }
        
        if self.transform:
            transformed = self.transform(**transform_input)
            image = transformed['image']
            transformed_boxes = transformed['bboxes']
            transformed_labels = transformed['category_ids']
        else:
            transformed_boxes = boxes
            transformed_labels = labels

        target = {}
        # Menangani kasus di mana semua bounding box hilang setelah augmentasi
        if len(transformed_boxes) > 0:
            # Konversi format [x,y,w,h] ke [x1,y1,x2,y2] yang dibutuhkan model
            boxes_xyxy = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in transformed_boxes]
            target["boxes"] = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed_labels, dtype=torch.int64)
            # Buat ulang area jika perlu (opsional, tergantung evaluator)
            target["area"] = torch.as_tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes_xyxy], dtype=torch.float32)
        else: 
            # Jika tidak ada box, buat tensor kosong untuk mencegah error
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["area"] = torch.zeros(0, dtype=torch.float32)

        target["image_id"] = torch.tensor([self.ids[idx]])
        target["iscrowd"] = torch.zeros((len(transformed_boxes),), dtype=torch.int64)

        return image, target
# --- PERUBAHAN SELESAI ---

def freeze_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
        if module.weight is not None: module.weight.requires_grad = False
        if module.bias is not None: module.bias.requires_grad = False

def main(args):
    model_base_name = args.model_base_name
    lr_str = str(args.lr).replace('.', '_')
    PROJECT_NAME = f"{model_base_name}_e{args.epochs}_bs{args.batch_size}_lr{lr_str}"
    DATASET_ROOT = 'dataset'
    CLASSES_TO_USE = ['baguette', 'cornbread', 'croissant', 'ensaymada', 'flatbread', 'sourdough', 'wheat-bread', 'white-bread', 'whole-grain-bread', 'pandesal']
    NUM_CLASSES = len(CLASSES_TO_USE)
    WORK_DIR = f"outputs/{PROJECT_NAME}"
    os.makedirs(WORK_DIR, exist_ok=True)
    print("="*60 + f"\nMemulai Eksperimen: {PROJECT_NAME}\n" + "="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BreadDataset(os.path.join(DATASET_ROOT, 'train'), os.path.join(DATASET_ROOT, 'train', 'filtered_annotations.coco.json'), transform=get_transform(train=True))
    dataset_test = BreadDataset(os.path.join(DATASET_ROOT, 'valid'), os.path.join(DATASET_ROOT, 'valid', 'filtered_annotations.coco.json'), transform=get_transform(train=False))
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = [m[0][0].in_channels for m in model.head.classification_head.module_list]
    new_head = SSDLiteClassificationHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=(NUM_CLASSES + 1), norm_layer=torch.nn.BatchNorm2d)
    model.head.classification_head = new_head
    model.to(device)
    
    # SARAN: Coba jalankan SATU KALI TANPA freeze_bn untuk melihat perbedaannya
    # model.apply(freeze_bn) 
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # --- AKTIFKAN KEMBALI LR SCHEDULER ---
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(f"--- MEMULAI TRAINING {args.epochs} EPOCH (DENGAN ALBUMENTATIONS & LR SCHEDULER) ---")
    best_map = 0.0
    training_history = []
    for epoch in range(args.epochs):
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        
        # Panggil scheduler setelah setiap epoch training
        lr_scheduler.step()
        
        eval_result = evaluate(model, data_loader_test, device=device)
        current_map = eval_result.coco_eval['bbox'].stats[0]
        # Catat learning rate yang sekarang digunakan
        current_lr = optimizer.param_groups[0]["lr"]
        training_history.append({'epoch': epoch + 1, 'loss': metric_logger.meters['loss'].global_avg, 'mAP_0.50:0.95': current_map, 'lr': current_lr})
        print(f"Epoch {epoch+1}/{args.epochs}: Avg Loss={metric_logger.meters['loss'].global_avg:.4f}, mAP={current_map:.4f}, LR={current_lr:.6f}")
        
        if current_map > best_map:
            best_map = current_map
            utils.save_on_master(model.state_dict(), os.path.join(WORK_DIR, 'best_model.pth'))
            print(f"*** Best mAP updated: {best_map:.4f} (model saved) ***")
    
    print(f"\n--- TRAINING SELESAI --- Best mAP: {best_map:.4f}")
    pd.DataFrame(training_history).to_csv(os.path.join(WORK_DIR, 'training_log.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-base-name', type=str, default="ssdmobilenetv3")
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)