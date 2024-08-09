#!/bin/bash

# DO NOT FORGET TO CONVERT UNIX (LF) USING NOTEPAD++

sudo apt-get update && sudo apt-get install

python3 -m virtualenv ~/env3_8
source ~/env3_8/bin/activate

pip install --upgrade pip

pip list

#Because of the bug that prevents nightly versions to be installed
pip install pip==23.*

pip list

# https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly+20240229-cp38-cp38-linux_x86_64.whl

pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.3.0.dev20240229%2Bcpu-cp38-cp38-linux_x86_64.whl

# https://github.com/pytorch/vision
# https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

pip install https://download.pytorch.org/whl/nightly/cpu/torchvision-0.18.0.dev20240229%2Bcpu-cp38-cp38-linux_x86_64.whl

# https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
# You can also add +yyyymmdd after torch_xla-nightly to get the nightly wheel of a specified date.
# To get the companion pytorch and torchvision nightly wheel, replace the torch_xla with torch or torchvision on above wheel links.

pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+20240229-cp38-cp38-linux_x86_64.whl

pip install https://storage.googleapis.com/libtpu-default-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20240229+default-py3-none-any.whl

pip list


cd ~/one-to-one-framework-frcpe/tpu_related/initializer_tpu_vm/
pip install -r requirements.txt
cd

pip list

pip install tensorflow-cpu tensorboard-plugin-profile
pip install --upgrade typing_extensions

pip list

echo "All Packages Are Installed"
