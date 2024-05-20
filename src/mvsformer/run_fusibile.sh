#!/bin/bash

# Define the base command for fusibile
base_cmd="./fusibile/fusibile"

# Define common options
common_opts="--depth_min=0.001 --depth_max=100000 --normal_thresh=360 --disp_thresh=0.5 --num_consistent=3.0 -color_processing"

# Array of folders to process
folders=("scan01" "scan02" "scan03" "scan04" "scan05" "scan06")

current_folder="1movement_cb/artificial"

# Loop through each folder and execute the command
for folder in "${folders[@]}"; do
    input_folder="exps/max/${current_folder}/${folder}/points_mvsnet/"
    p_folder="${input_folder}cams/"
    images_folder="${input_folder}images/"
    
    # Construct the full command with arguments
    cmd="$base_cmd -input_folder $input_folder -p_folder $p_folder -images_folder $images_folder $common_opts"
    
    # Execute the command
    echo "Processing $folder..."
    $cmd
done

echo "All processing complete."