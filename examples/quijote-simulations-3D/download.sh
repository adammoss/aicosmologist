#!/bin/bash
set -e

# Create the data directory if it doesn't exist
DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Download the files using wget
wget -O "$DATA_DIR/latin_hypercube_3D.npy" "https://storage.googleapis.com/quijote-simulations/latin_hypercube_3D.npy"
wget -O "$DATA_DIR/latin_hypercube_params.txt" "https://storage.googleapis.com/quijote-simulations/latin_hypercube_params.txt"

echo "Downloads complete."

