#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the Kaggle CLI is installed.
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI is not installed. Please install it first." >&2
    exit 1
fi

# Create the data directory if it doesn't exist.
mkdir -p data

# Change to the data directory.
cd data

# Download the competition data.
kaggle competitions download -c galaxy-zoo-the-galaxy-challenge

echo "Download complete. Unzipping the main competition zip file..."

# Unzip the downloaded competition zip file.
unzip -q galaxy-zoo-the-galaxy-challenge.zip

# Remove the main zip file to free up space.
rm galaxy-zoo-the-galaxy-challenge.zip

# Delete the benchmark zip files.
rm central_pixel_benchmark.zip

# Unzip the remaining zip files.
unzip -q images_test_rev1.zip
unzip -q images_training_rev1.zip
unzip -q training_solutions_rev1.zip
unzip -q all_ones_benchmark.zip
unzip -q all_zeros_benchmark.zip

# Delete the zip files after unzipping.
rm images_test_rev1.zip images_training_rev1.zip training_solutions_rev1.zip

# Rename directories and file.
mv images_test_rev1 images_test
mv images_training_rev1 images_training
mv training_solutions_rev1.csv training_solutions.csv

echo "All files have been unzipped and organized in the data directory."
