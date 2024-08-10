This repo contains Python scripts from [OANET](https://github.com/zjhthu/OANet) repo.

The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1j5z-FdzlgzZMB9qxcNyps2j61PjLYnqY?usp=sharing).

## Download OANET Repo and YFCC Dataset

Clone the [OANET](https://github.com/zjhthu/OANet) repo to your local machine.

Download [YFCC Dataset](https://drive.google.com/drive/folders/1xrc6ZuCOGYwno1DEIfK-jbvZGqK4Oc79) into cloned OANET folder.

Generate raw_data folder in cloned OANET folder with:
```
cat raw_data* > combined_file.tar.gz
tar -xvzf combined_file.tar.gz
```
## Generate .hdf5 Files

Generate .hdf5 Files using the scripts in OANET folder:
```
cd dump_match
python extract_feature.py
python yfcc.py
```
This may take a couple of hours.

## Generate pickle files from .hdf5 Files

Clone the current [one-to-one-framework-frcpe](https://github.com/mfatih7/one-to-one-framework-frcpe) repo to your local machine.

Generate a folder named `01_datasets` to the same directory level with the cloned `one-to-one-framework-frcpe` repo folder.

Copy previously generated `yfcc-sift-2000-train.hdf5`, `yfcc-sift-2000-val.hdf5`, and `yfcc-sift-2000-test.hdf5` files into `01_datasets` folder.

To generate pickle files, run

```
cd one-to-one-framework-frcpe
python convertHDF5toPickle.py
```

This may take a couple of hours.
The script generates 3 sets of pickle files.

For each of train, validation, and test operations:
- 8 pickle files are generated for set 2,
- 4 pickle files are generated for set 1,
- 1 pickle file is generated for set 0.

The sets are proper for different TPU operations.

- Set 2 is proper for multi-core TPUv2 and TPUv3 operations.
- Set 1 is proper for multi-core TPUv4 operations.
- Set 0 is proper for SPMD and single-core operations.

For GPU operations, a set can be chosen with respect to the main memory capacity of the machine.

## Generate Google Cloud TPU Virtual Machine and Cloud Storage Bucket

Explore the [experiments.txt](https://drive.google.com/drive/folders/1jcBMZOKO3KTIlhfHuFwYWJCSY2RIYuKO) to determine the TPU version you need. TPUv4 is sufficient for all model types.

Obtain a TPU Virtual Machine (TPU-VM) and Google Storage Bucket. Maybe [TPU Research Cloud](https://sites.research.google/trc/about/) can be helpful.

To generate a TPU VM using Google Cloud CLI, the script [generate_tpu_vm.py](https://github.com/mfatih7/one-to-one-framework-frcpe/blob/main/tpu_related/generate_tpu_vm/generate_tpu_vm.py) can be used.

## Copy the Generated Dataset Folder into the Bucket

Copy the `01_datasets` folder into Bucket using the Bucket GUI or Google Cloud CLI.

You can also copy the folder into each TPU-VM, but Buckets are more manageable and storage-friendly.

## Clone This Repo into TPU-VM and Generate Python Environment

Clone this Repo into TPU-VM using

```
git clone https://github.com/mfatih7/one-to-one-framework-frcpe.git
```

Use the script below to generate an environment in which all of the networks are trained in this study.

```
bash ~/one-to-one-framework-frcpe/tpu_related/initializer_tpu_vm/init_tpu_vm_nightly.sh
```

This environment (Python 3.8) does not have TPU-lowered implementations of [some functions](https://github.com/pytorch/xla/issues/6017).

Use the script below to generate a newer environment (with Python 3.10, nightly torch, torchvision, and torch_xla) that does not have lowering issues.

```
bash ~/one-to-one-framework-frcpe/tpu_related/initializer_tpu_vm/init_tpu_vm_nightly_latest.sh
```

Be aware that [PyTorch/XLA](https://github.com/pytorch/xla) is still being developed, be careful about the updates.

## Testing the Pre-Trained Models

...

## Training Your Own Models

...








