#!/bin/bash

mkdir data_storage/RICO_dataset

TARGET_DIR="data_storage/RICO_dataset"

if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    echo "Folder $TARGET_DIR is not empty!"
    exit 1 
fi
sudo apt install rsync pv -y

# Install RICO dataset
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
tar -xvzf unique_uis.tar.gz
find combined/ -mindepth 1 -maxdepth 1 | pv -lep -s $(find combined/ -mindepth 1 -maxdepth 1 | wc -l) | xargs -I {} mv "{}" data_storage/RICO_dataset/
rm -r unique_uis.tar.gz unique_uis
