#!/usr/bin/env bash

unzip /group/30042/zhouxiawang/env/share/lib_package/ninja-linux.zip -d /usr/local/bin/
update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1


pip install facexlib==0.2.5
# pip install ffmpeg-python==0.2.0
pip install omegaconf==2.0.6
pip install pytorch-lightning==1.0.8
pip install basicsr==1.3.3.4
pip install test-tube==0.7.5
pip uninstall opencv-python
pip install opencv-python==4.5.2.54

pip list

# -----------------------
# copy model
# -----------------------
# mkdir /usr/local/app/.local/lib/python3.8/site-packages/facexlib/weights
if [ ! -d "/ /root/" ];then
  mkdir /root
fi
if [ ! -d "/ /root/.cache" ];then
  mkdir /root/.cache
fi
if [ ! -d "/root/.cache/torch" ];then
  mkdir /root/.cache/torch
fi
if [ ! -d "/root/.cache/torch/hub" ];then
  mkdir /root/.cache/torch/hub
fi
if [ ! -d "/root/.cache/torch/hub/checkpoints" ];then
  mkdir /root/.cache/torch/hub/checkpoints
fi

ls -l /group/30042/zhouxiawang/env/share/weights/facexlib
ls -l /group/30042/zhouxiawang/env/share/weights/
ln -s /group/30042/zhouxiawang/env/share/weights/facexlib /usr/local/app/.local/lib/python3.8/site-packages/facexlib/weights
cp /group/30042/zhouxiawang/env/share/weights/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints
cp /group/30042/zhouxiawang/env/share/weights/pt_inception-2015-12-05-6726825d.pth  /root/.cache/torch/hub/checkpoints
ls -l "/root/.cache/torch/hub/checkpoints"
