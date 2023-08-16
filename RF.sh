#!/usr/bin/env bash

source /data/miniconda3/etc/profile.d/conda.sh
which conda
export PATH=/usr/local/app/.local/bin:$PATH

# -----------------------
# activate conda env
# -----------------------
# conda init bash
# conda deactivate
which python
source activate pt19
which python
# PYTHON=${PYTHON:-"/data/miniconda3/envs/pt19/bin/python"}

# # -----------------------
# # activate gcc 7.3
# # -----------------------
# scl enable devtoolset-7 bash
# source /opt/rh/devtoolset-7/enable
# gcc --version

# -----------------------
# install ninja-linux
# -----------------------
sudo unzip /group/30042/zhouxiawang/env/share/lib_package/ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1

# # -----------------------
# # show info
# # -----------------------
nvidia-smi
cat /etc/issue
$PYTHON -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version()); print(torch.cuda.is_available())"

# -----------------------
# run script
# -----------------------
cd /group/30042/zhouxiawang/project/release/RestoreFormer
pwd
which pip
which python
# sudo pip install -r RF_requirements.txt

sudo pip install facexlib==0.2.5
# sudo pip install ffmpeg-python==0.2.0
sudo pip install omegaconf==2.0.6
sudo pip install pytorch-lightning==1.0.8
sudo pip install basicsr==1.3.3.4
sudo pip install test-tube==0.7.5
sudo pip uninstall opencv-python
sudo pip install opencv-python==4.5.2.54

sudo pip list

# -----------------------
# copy model
# -----------------------
# sudo mkdir /usr/local/app/.local/lib/python3.8/site-packages/facexlib/weights
if [ ! -d "/ /root/" ];then
  sudo mkdir /root
fi
if [ ! -d "/ /root/.cache" ];then
  sudo mkdir /root/.cache
fi
if [ ! -d "/root/.cache/torch" ];then
  sudo mkdir /root/.cache/torch
fi
if [ ! -d "/root/.cache/torch/hub" ];then
  sudo mkdir /root/.cache/torch/hub
fi
if [ ! -d "/root/.cache/torch/hub/checkpoints" ];then
  sudo mkdir /root/.cache/torch/hub/checkpoints
fi

sudo ls -l /group/30042/zhouxiawang/env/share/weights/facexlib
sudo ls -l /group/30042/zhouxiawang/env/share/weights/
sudo cp /group/30042/zhouxiawang/env/share/weights/facexlib/*.pth /usr/local/app/.local/lib/python3.8/site-packages/facexlib/weights/
sudo cp /group/30042/zhouxiawang/env/share/weights/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints
sudo ls -l "/root/.cache/torch/hub/checkpoints"
sudo ls -l "/usr/local/app/.local/lib/python3.8/site-packages/facexlib/weights/"

export OMP_NUM_THREADS=6

export BASICSR_JIT=True

root_path='/group/30042/zhouxiawang/checkpoints/RestoreFormer/release'

conf_name='HQ_Dictionary'
# conf_name='RestoreFormer'
conf_name='ROHQD'

gpus='0,'
# gpus='0,1,2,3,4,5,6,7'
node_n=1
ntasks_per_node=1
gpu_n=$(expr $node_n \* $ntasks_per_node)

# python -u main.py \
sudo python -m torch.distributed.launch --nproc_per_node=$ntasks_per_node main.py \
--root-path $root_path \
--base 'configs/'$conf_name'.yaml' \
-t True \
--postfix $conf_name'_lmdb_gpus'$gpu_n'_h4_E62' \
--gpus $gpus \
--num-nodes $node_n \
--random-seed True \
2>&1 |tee $root_path'/'$conf_name'_lmdb_gpus'$gpu_n'_h4_E62_log_v1.txt'
