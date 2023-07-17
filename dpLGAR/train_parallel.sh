#!/bin/bash

# Check if the file path is provided as an argument
if [ $# -lt 1 ]; then
    echo "Please provide the path to the file containing basin_ids."
    exit 1
fi

file_path=$1
mapfile -t basin_ids < "$file_path"

# Loop through each basin_id and call dpLGAR
for basin_id in "${basin_ids[@]}"; do
    python -m dpLGAR --multirun "++basin_id=$basin_id"
done