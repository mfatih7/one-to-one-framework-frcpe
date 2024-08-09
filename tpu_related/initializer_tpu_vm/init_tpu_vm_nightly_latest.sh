#!/bin/bash

# DO NOT FORGET TO CONVERT UNIX (LF) USING NOTEPAD++

# Update package list and install necessary packages for Python 3.10
sudo apt-get update && sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create a virtual environment using Python 3.10
python3.10 -m venv ~/env3_10
source ~/env3_10/bin/activate

pip install --upgrade pip

pip list

#Because of the bug that prevents nightly versions to be installed
pip install pip==23.*

pip list

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-releases/index.html

pip list

cd ~/one-to-one-framework-frcpe/tpu_related/initializer_tpu_vm/
pip install -r requirements.txt
cd

pip list

pip install tensorflow-cpu tensorboard-plugin-profile
pip install --upgrade typing_extensions

pip list

echo "All Packages Are Installed"
