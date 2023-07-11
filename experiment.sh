#!/bin/bash

# Specify the path to the requirements.txt file
requirements_file="requirements.txt"

# Check if requirements.txt file exists
if [ ! -f "$requirements_file" ]; then
    echo "Error: requirements.txt file not found."
    exit 1
fi

pip install lightly

# Read each line in requirements.txt and install the packages
while read -r package; do
    # Ignore comments and empty lines
    if [[ $package != \#* ]] && [ -n "$package" ]; then
        echo "Installing package: $package"
        pip install "$package"
    fi
done < "$requirements_file"

# Run the experiment
python3 pipeline.py --pretrain_epochs 1 --supervised_epochs 0 --no_batchnorm
python3 pipeline.py --pretrain_epochs 1 --supervised_epochs 0 --Adam
python3 pipeline.py --pretrain_epochs 1 --supervised_epochs 0 --no_scheduler
python3 pipeline.py --pretrain_epochs 1 --supervised_epochs 0 --output_dim 256
python3 pipeline.py --pretrain_epochs 1 --supervised_epochs 0 --lower_precision
