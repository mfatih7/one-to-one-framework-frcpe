This repo contains Python scripts from [OANET](https://github.com/zjhthu/OANet) repo.

The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1j5z-FdzlgzZMB9qxcNyps2j61PjLYnqY?usp=sharing).

## Download OANET Repo and YFCC Dataset

Clone the [OANET](https://github.com/zjhthu/OANet) repo.

Download [YFCC Dataset](https://drive.google.com/drive/folders/1xrc6ZuCOGYwno1DEIfK-jbvZGqK4Oc79) into cloned OANET folder.

Generate raw_data folder in cloned OANET folder with.
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

Clone the current [one-to-one-framework-frcpe](https://github.com/mfatih7/one-to-one-framework-frcpe) repo.

Generate a folder named `01_datasets` to the same directory level with the cloned `one-to-one-framework-frcpe` repo folder.

Copy previously generated `yfcc-sift-2000-train.hdf5`, `yfcc-sift-2000-val.hdf5`, and `yfcc-sift-2000-test.hdf5` files into `01_datasets` folder.

To generate pickle files, run

```
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

Check out the [experiments.txt](https://drive.google.com/drive/folders/1jcBMZOKO3KTIlhfHuFwYWJCSY2RIYuKO) to determine the TPU version you need. TPUv4 is sufficient for all.

Obtain a TPU Virtual Machine (TPU-VM) and Google Storage Bucket. Maybe [TPU Research Cloud](https://sites.research.google/trc/about/) can be helpful.

## Copy the Generated Dataset Files into the Bucket

...

## Clone This Repo into TPU-VM and Generete Python Environment

...




