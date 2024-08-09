#!/bin/bash

# DO NOT FORGET TO CONVERT UNIX (LF) USING NOTEPAD++

sudo apt-get update && sudo apt-get install -y python3-opencv

python3 -m virtualenv ~/env3_8
source ~/env3_8/bin/activate

pip install --upgrade pip

pip list

#Because of the bug that prevents nightly versions to be installed
pip install pip==23.*

pip list

#pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl

pip list

cd ~/one-to-one-framework-frcpe/tpu_related/initializer_tpu_vm/
pip install -r requirements.txt
cd

pip list

pip install tensorflow-cpu tensorboard-plugin-profile
pip install --upgrade typing_extensions

pip list

echo "All Packages Are Installed"
