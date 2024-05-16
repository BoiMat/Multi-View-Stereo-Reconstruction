#!/bin/bash

# Base directory where the 'colmap' folder is located
base_dir="colmap"
output_dir="deep"
# List of movements
movements=("1movement_cb" "1movement_nocb" "2movements_cb")
# List of light conditions
conditions=("artificial" "natural")

for movement in "${movements[@]}"; do
    for condition in "${conditions[@]}"; do
        current_dir="$base_dir/$movement/$condition"
        for dir_name in "$current_dir"/*; do
            if [ -d "$dir_name" ]; then
                dense_folder="$dir_name/dense"
                output_folder="$output_dir/$movement/$condition/$(basename "$dir_name")"
                mkdir -p "$output_folder"

                command="python colmap2mvsnet.py --dense_folder \"$dense_folder\" --output_folder \"$output_folder\" --modified"
                echo "$command"
                eval "$command"
            fi
        done
    done
done