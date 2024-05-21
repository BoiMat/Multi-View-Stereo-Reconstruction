import cv2
import numpy as np
import os
from colmap2mvsnet import rotmat2qvec, qvec2rotmat
from argparse import ArgumentParser
from utils import *

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    
    directory_path = args.data_path

    file_name = "calibrated_camera.yml"
    file = os.path.join(directory_path, file_name)
    intrinsics, dist_coeffs = load_intrinsics(file)

    directories = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
    if 'objects' in directories:
        directories.remove('objects')
    if 'textured_objects' in directories:
        directories.remove('textured_objects')
    directories = sorted(directories)
        

    for directory in directories:
        folders_path = os.path.join(directory_path, directory)
        for category in ['artificial', 'natural']:
            category_path = os.path.join(folders_path, category)
            
            # load the reference cameras from the file
            file_name = "reference_cameras.txt"
            file_path = os.path.join(category_path, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                best_pairs = [[int(p) for p in pair.split()] for pair in lines]
            
            folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            for i,folder in enumerate(folders):
                print(f'Processing {directory}/{category}/{folder}..')
                obj_dir = os.path.join(category_path, folder)
                
                file_name = "pose.yml"
                file_path = os.path.join(obj_dir, file_name)
                camera_poses = load_camera_poses(file_path)
                
                undistorted_path = obj_dir+'/undistorted_images'
                images = load_images(obj_dir+'/images', camera_poses, undistorted_path)
                
                best_pair = best_pairs[i]
                        
                # save the indices of the two ref images
                ref_R0 = images[best_pair[0]]['R']
                ref_T0 = images[best_pair[0]]['T']

                ref_R1 = images[best_pair[1]]['R']
                ref_T1 = images[best_pair[1]]['T']
                
                for i, image in images.items():
                    R_relative, T_relative = compute_relative_pose_to_midpoint(ref_R0, 
                                                                ref_T0,
                                                                ref_R1,
                                                                ref_T1, 
                                                                image['R'],
                                                                image['T'])
                    
                    image['R_relative'] = R_relative
                    image['T_relative'] = T_relative
                    
                    R = R_relative.T
                    T = - R @ T_relative
                    # meters to millimeters
                    # T *= 1000

                    qvec = rotmat2qvec(R)
                    image['colmap'] = (qvec, T)
                
                current_path = os.path.join(args.output_path, directory, category, folder)
                save_colmap_dataset(images, current_path, intrinsics, dist_coeffs)