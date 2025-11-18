git clone https://github.com/open-mmlab/mmdetection.git

1. install anaconda
2. buka anaconda
3. ikuti step dibawah ini
4. masuk ke train.ipynb, di kanan atas ada pilihan kernel cari roti_env

```bash
# 1. Buat environment
conda create -n roti_env python=3.10 
conda activate roti_env

# 2. Install PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
pip install -U openmim
mim install mmdet
pip install future tensorboard
pip install roboflow pandas seaborn matplotlib
pip install -U mmpretrain
conda install numpy scikit-learn jupyter cmake ninja 
# 4. Jalankan Jupyter Notebook
jupyter notebook



```

scheduler apapun 49-50%

statis lr 0.002 batch size 8 epoch 30 = 50%
statis lr 0.004 batch size 16 epoch 30 = 50%

| Skenario | LR    | Batch Size | Image Size | Epoch | Akurasi | Waktu Train |
|----------|-------|------------|------------|--------|----------|--------------|
| 1        |0.001  |4           |320         |30      |48%       |23 Menit      |
| 2        |0.001  |4           |320         |50      |50%       |32 Menit      |
| 3        |0.001  |4           |320         |70      |50%       |50 menit      |
| 4        |0.001  |4           |320         |100     |51%       |70 Menit      |
| 5        |0.001  |8           |320         |30      |48%       |20 Menit      |
| 6        |0.001  |8           |320         |50      |50%       |31 Menit      |
| 7        |0.001  |8           |320         |70      |50%       |47 Menit      |
| 8        |0.001  |8           |320         |100     |51%       |67 Menit      |
| 9        |0.001  |16          |320         |30      |44%       |20 Menit      |
| 10       |0.001  |16          |320         |50      |48%       |31 Menit      |
| 11       |0.001  |16          |320         |70      |49%       |45 Menit      |
| 12       |0.001  |16          |320         |100     |50%       |64 menit      |



