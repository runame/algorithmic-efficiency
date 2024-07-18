# Download data sets that are not on the vector cluster, link the others
mkdir data

# Need to download OGBG
python3 datasets/dataset_setup.py --data_dir data/ --ogbg

# symlink WMT
ln -s /datasets/wmt/ ./data/wmt

# symlink ImageNet
mkdir data/imagenet
ln -s /datasets/imagenet/train data/imagenet/train
ln -s /datasets/imagenet/val data/imagenet/val
# download ImageNetV2 (less than 2 GiB)
python download_imagenet_v2.py --data-dir=data/imagenet

# symlink FastMRI
mkdir data/fastmri
ln -s /datasets/fastMRI/singlecoil_test ./data/fastmri/knee_singlecoil_test
ln -s /datasets/fastMRI/singlecoil_train ./data/fastmri/knee_singlecoil_train
ln -s /datasets/fastMRI/singlecoil_val ./data/fastmri/knee_singlecoil_val
