import os
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default="../deep")
parser.add_argument("--out_path", type=str, default="exps")
args = parser.parse_args()

# Base command template
base_command = ("python test.py --dataset custom --batch_size 1 "
                "--testpath {testpath} "
                "--testlist ./lists/custom/test.txt "
                "--resume ./pretrained_weights/MVSFormer/best.pth "
                "--outdir {outdir} "
                "--fusibile_exe_path ./fusibile/fusibile "
                "--interval_scale 2.0 --num_view 5 "
                "--numdepth 512 --max_h 1024 --max_w 1280 --filter_method gipuma "
                "--disp_threshold 0.5 --num_consistent 3 "
                "--prob_threshold 0.5,0.5,0.5,0.5 "
                "--combine_conf --tmps 5.0,5.0,5.0,1.0")

# Fusibile command template
fusibile_command = ("./fusibile/fusibile -input_folder {input_folder} "
                    "-p_folder {p_folder} -images_folder {images_folder} "
                    "--depth_min=0.001 --depth_max=100000 --normal_thresh=360 "
                    "--disp_thresh=0.5 --num_consistent=3.0 -color_processing")

# Folders and categories to iterate over
folders = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]

for folder in folders:
    categories = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    for category in categories:
        
        testpath = f"{args.data_path}/{folder}/{category}"
        outdir = f"{args.out_path}/{folder}/{category}"
        
        # Format the command with the current testpath and outdir
        command = base_command.format(testpath=testpath, outdir=outdir)
        
        # Execute the first command
        print(f"Executing first command for {folder}/{category}")
        subprocess.run(command, shell=True)
        # print(command)

        # Now run the fusibile command for each scan in the outdir
        scans = [name for name in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, name))]
        for scan in scans:
            input_folder = f"{outdir}/{scan}/points_mvsnet/"
            p_folder = f"{outdir}/{scan}/points_mvsnet/cams/"
            images_folder = f"{outdir}/{scan}/points_mvsnet/images/"
            # Format the fusibile command with the current folders
            fusibile_cmd = fusibile_command.format(input_folder=input_folder, p_folder=p_folder, images_folder=images_folder)
            
            # Execute the fusibile command
            print(f"Executing fusibile command for {folder}/{category}/{scan}")
            subprocess.run(fusibile_cmd, shell=True)
            # print(fusibile_cmd)
