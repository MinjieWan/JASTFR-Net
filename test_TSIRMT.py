import torchvision.transforms as transforms
import gc
import time
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from JASTFRNet import JASTFRNet
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from data_loader_TSIRMT import TestSequenceDataset


def test_all_sequences(test_root='./dataset/TSIRMT/val',
                       output_root='./dataset/TSIRMT/val_output',
                       ckpt_path='./checkpoints/TSIRMT.pth',
                       resize_to_256=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = JASTFRNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    img_root = os.path.join(test_root, 'images')

    # Collect all sequence directories
    seq_dirs = [os.path.join(img_root, d)
                for d in sorted(os.listdir(img_root))
                if os.path.isdir(os.path.join(img_root, d))]

    if not seq_dirs:
        raise RuntimeError('No sequence found under test_root')

    all_preds = []
    all_gts = []

    total_inference_time = 0.0  # Total inference time (seconds)
    total_frames = 0  # Total frames

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        print(f'\n==== Processing {seq_name} ====')

        # dataset
        dataset = TestSequenceDataset(seq_dir, seq_len=5, resize_to_256=resize_to_256)

        # Output Directory
        out_img_dir = os.path.join(output_root, seq_name)
        os.makedirs(out_img_dir, exist_ok=True)

        preds_seq = []
        gts_seq = []

        seq_inference_time = 0.0  # Total time for single sequence
        seq_frames = 0  # Number of frames per single sequence

        with torch.no_grad():
            for seq, mask_gt, mask_path in tqdm(dataset, desc=seq_name):
                seq = seq.unsqueeze(2)
                seq = seq.permute(0, 2, 1, 3, 4)
                seq = seq.to(device)

                # ================== Timing starts ==================
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()

                pred = net(seq)  # (1,1,H,W)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                # ================== Timing ends ==================

                t = end - start
                print(f"t: {t:.4f}")

                # Cumulative time statistics
                seq_inference_time += t
                seq_frames += 1

                # 1. Calculate the sigmoid probability value (0~1)
                pred_prob = torch.sigmoid(pred)

                # 2. Binarization results (used for subsequent metric calculation, stored in preds_seq)
                pred_binary = pred_prob > 0.5
                pred_binary_np = pred_binary.cpu().numpy().astype(np.uint8)[0, 0]

                # 3. Grayscale image result (for saving images)
                pred_gray_np = pred_prob.cpu().numpy()[0, 0]
                pred_gray_np = (pred_gray_np * 255).astype(np.uint8)

                mask_np = mask_gt.cpu().numpy().astype(np.uint8)[0, 0]

                # The binarization results are stored in the sequence
                preds_seq.append(pred_binary_np)
                gts_seq.append(mask_np)

                # What is saved is a grayscale image
                frame_id = os.path.basename(mask_path)
                cv2.imwrite(os.path.join(out_img_dir, frame_id), pred_gray_np)

        # Accumulate to global statistics
        total_inference_time += seq_inference_time
        total_frames += seq_frames

        # Calculate the average time consumption of a single sequence
        avg_time_per_frame = seq_inference_time / seq_frames if seq_frames > 0 else 0

        # Calculate single sequence metrics
        pred_flat = np.concatenate([p.ravel() for p in preds_seq]).astype(bool)
        gt_flat = np.concatenate([g.ravel() for g in gts_seq]).astype(bool)

        iou = np.sum(pred_flat & gt_flat) / np.sum(pred_flat | gt_flat)
        dice = f1_score(gt_flat, pred_flat)
        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)

        print(f'{seq_name}  Frames: {seq_frames}  AvgTime: {avg_time_per_frame * 1000:.2f}ms  '
              f'IoU: {iou:.4f}  Dice: {dice:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}')

        all_preds.append(pred_flat)
        all_gts.append(gt_flat)

    # Global indicators
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    global_iou = np.sum(all_preds & all_gts) / np.sum(all_preds | all_gts)
    global_dice = f1_score(all_gts, all_preds)
    global_prec = precision_score(all_gts, all_preds, zero_division=0)
    global_rec = recall_score(all_gts, all_preds, zero_division=0)
    global_avg_time = total_inference_time / total_frames if total_frames > 0 else 0

    print('\n==== Global Results ====')
    print(f'Total Frames: {total_frames}  Total Time: {total_inference_time:.2f}s')
    print(f'Global Avg Inference Time: {global_avg_time * 1000:.2f}ms ({global_avg_time:.4f}s)')
    print(f'Global IoU: {global_iou:.4f}  Dice: {global_dice:.4f}  '
          f'Precision: {global_prec:.4f}  Recall: {global_rec:.4f}')
    print(f'All outputs saved to {output_root}')


if __name__ == '__main__':
    test_all_sequences()
