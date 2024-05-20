import os
import yaml
import subprocess
from argparse import ArgumentParser

# Define the argument parser
parser = ArgumentParser()
parser.add_argument("--base_yaml_path", type=str, required=True)
parser.add_argument("--base_directory", type=str, required=True)
parser.add_argument("--output_base", type=str, required=True)
args = parser.parse_args()

base_yaml_path = args.base_yaml_path
base_directory = args.base_directory
output_base = args.output_base

# Loop through each folder in the base directory
for folder in os.listdir(base_directory):
    for category in ['artificial', 'natural']:
        folder_path = os.path.join(base_directory, folder, category)
        if os.path.isdir(folder_path):
            # Load the base YAML configuration
            with open(base_yaml_path, 'r') as file:
                config = yaml.safe_load(file)

            # Update the configuration
            current_output_dir = os.path.join(output_base, folder, category)
            current_root_dir = os.path.join(base_directory, folder, category)
            config["output_dir"] = current_output_dir
            config["data"]["test"]["root_dir"] = current_root_dir
            config["fusion"]["xy_filter_per"]["output_dir"] = current_output_dir

            # Save the updated configuration to a new file
            updated_config_filename = f"configs/test_{folder}_{category}.yaml"
            with open(updated_config_filename, 'w') as file:
                yaml.dump(config, file, default_flow_style=None)

            # Run the command with the newly created .yaml file
            command = f"python test_gbinet.py --cfg {updated_config_filename}"
            subprocess.run(command, shell=True)
            print(command)