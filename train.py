import torch
import torch.nn as nn
import torch.nn.functional as F
from JASTFRNet import JASTFRNet, HybridLoss
from data_loader_TSIRMT import SequenceDataset
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = JASTFRNet()
    net.to(device)

    # Parameter Configuration
    # ---------------------------------
    batch_size = 4

    train_dataset = SequenceDataset('./dataset/TSIRMT', mode='train', resize_to_256=False)
    val_dataset = SequenceDataset('./dataset/TSIRMT', mode='val', resize_to_256=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion = HybridLoss(pos_weight=5.0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-5
    )

    num_epochs = 100
    best_IoU = 0.0
    best_model_path = './checkpoints/'
    # ---------------------------------

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            outputs = outputs.squeeze()
            labels = labels.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            torch.cuda.empty_cache()
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # validate
        if (epoch + 1) % 10 == 0:
            net.eval()
            running_loss = 0.0
            all_pred = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    outputs = outputs.squeeze()
                    labels = labels.squeeze()
                    loss = criterion(outputs, labels)

                    outputs = torch.sigmoid(outputs)
                    outputs = outputs.cpu().numpy()
                    labels = labels.cpu().numpy()

                    preds = (outputs > 0.5).astype(np.float32)

                    all_pred.append(preds.flatten())
                    all_labels.append(labels.flatten())

                    running_loss += loss.item()

            val_loss = running_loss / len(val_loader)

            pred = np.concatenate(all_pred).astype(bool)
            gt = np.concatenate(all_labels).astype(bool)

            iou = np.sum(pred & gt) / np.sum(pred | gt)
            dice = f1_score(gt, pred)
            precision = precision_score(gt, pred)
            recall = recall_score(gt, pred)

            # Print training and validation results
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val iou: {iou:.4f}, '
                  f'Val dice: {dice:.4f}, '
                  f'Val precision: {precision:.4f}, '
                  f'Val recall: {recall:.4f} '
                  )

            if iou > best_IoU:
                best_IoU = iou
                pth_name = 'TSIRMT_Epoch-%3d_iou-%.4f.pth' % (epoch + 1, best_IoU)
                torch.save(net.state_dict(), best_model_path + pth_name)
            print(f"Epoch {epoch+1}, Best model saved with iou: {best_IoU:.4f}")

    print("Training finished")


if __name__ == "__main__":
    main()

