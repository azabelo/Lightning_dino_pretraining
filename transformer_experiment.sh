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

python3 transformer_pipeline.py --pretrain_epochs 0 --supervised_epochs 1




