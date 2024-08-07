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

Generate a folder named `01_featureMatchingDatasets` to the same directory level with the cloned `one-to-one-framework-frcpe` repo folder.

Copy the generated `yfcc-sift-2000-train.hdf5`, `yfcc-sift-2000-val.hdf5`, and `yfcc-sift-2000-test.hdf5` files into `01_featureMatchingDatasets` folder.





