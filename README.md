# JASTFRNet

Official PyTorch implementation of JASTFRNet.

---

## ⚙️ Requirements

- Python 3.8  
- PyTorch 1.12.0 (CUDA 11.3)  
- torchvision 0.13.0  
- opencv-python  
- numpy  
- scikit-image  
- pytorch-msssim  

---

## 📂 Dataset

We use the following datasets:

### NUDT-MIRSDT
- Download: https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA  
- Code: `5whn`

### TSIRMT
- Download: https://drive.google.com/drive/folders/1aWDNdUWkTOuV3fILbgLDEqM2N2erW05n  

⚠️ **Note:**
- The `dataset/` folder in this repository only contains **partial samples**.  
- Please download the **full datasets manually** from the links above.

---

## 📁 Project Structure

```bash
JASTFRNet/
├── dataset/        # sample data (not full dataset)
├── checkpoints/    # trained model
├── JASTFRNet.py    # network definition
├── train.py        # training script
└── test_xxx.py     # testing script
