git clone https://github.com/open-mmlab/mmdetection.git


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

