import os
import torch
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
from argparse import ArgumentParser
from utils import *

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)


def preprocess_images(path, intrinsics, dist_coeffs):
    
    file_name = "pose.yml"
    file_path = os.path.join(path, file_name)
    camera_poses = load_camera_poses(file_path)
    
    # if the undistorted images already exist, load them
    undistorted_path = path+'/undistorted_images'
    if os.path.exists(undistorted_path):
        images = load_images(path+'/images', camera_poses, undistorted_path)
    # otherwise, load the images and undistort them
    else:
        images = load_images(path+'/images', camera_poses)
        undistort_images(images, intrinsics, dist_coeffs, path+'/undistorted_images')
        
    return images


def best_pair_selection(images):
    # compute the scores to find the best pair of images
    view_sel = computer_scores(images)
    
    best_pair = None
    max_matches = 0
    for j in range(len(view_sel)):
        if view_sel[j][0][1] > max_matches:
            max_matches = view_sel[j][0][1]
            best_pair = (j, view_sel[j][0][0])
    
    return best_pair


def load_mesh(obj_filename, device, scaling_factor=0.001):
    mesh = load_objs_as_meshes([obj_filename], device=device)
    scaled_vertices = mesh.verts_packed() * scaling_factor
    mesh = mesh.update_padded(scaled_vertices.unsqueeze(0))
    
    if not mesh.textures:
        print("WARNING: failed to load textures. Using dummy textures. (Normal for original meshes)")
        verts_shape = mesh.verts_packed().shape
        blue_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # RGB blue
        dummy_texture = blue_color.repeat(1, verts_shape[0], 1)
        mesh.textures = TexturesVertex(verts_features=dummy_texture)
    
    return mesh


def set_renderer(mesh, image, intrinsics, device):
    
    min_vals, _ = torch.min(mesh.verts_packed(), dim=0)
    image_height, image_width = image['undistorted_image'][:,:,0].shape
    
    R_adjusted, T_final = checkerboard_to_view(image['R'], image['T'], min_vals.cpu().numpy())
    R = torch.Tensor(R_adjusted).unsqueeze(0).to(device)
    T = torch.Tensor(T_final).unsqueeze(0).to(device)

    image_size = ((image_height, image_width),)
    fcl_screen = ((intrinsics[0,0], intrinsics[1,1]),)
    prp_screen = ((intrinsics[0,2], intrinsics[1,2]),)

    camera = PerspectiveCameras(R=R,
                                T=T,
                                focal_length=fcl_screen,
                                principal_point=prp_screen,
                                in_ndc=False,
                                image_size=image_size,
                                device=device)
    
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
        )
    )

    return renderer


def generate_ground_truth_pointcloud(mesh, image_pair, num_samples=100000, height_threshold=0.0015):
    
    points, features = sample_points_from_meshes(mesh, num_samples=num_samples, return_textures=True)
    gt_pointcloud = Pointclouds(points=points, features=features)
    # Convert Open3D PointCloud to NumPy array for easier manipulation
    pcd_points = gt_pointcloud.points_packed().cpu().numpy()

    # Select points above the height threshold
    points_above_threshold = pcd_points[pcd_points[:, 2] > height_threshold]
    
    ref_R0 = image_pair[0]['R']
    ref_T0 = image_pair[0]['T']

    ref_R1 = image_pair[1]['R']
    ref_T1 = image_pair[1]['T']
    
    R_cb_to_torch = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    min_vals, _ = torch.min(mesh.verts_packed(), dim=0)
    O_cb_to_torch = min_vals.cpu().numpy()  # translation vector from checkerboard to torch origin

    T_midpoint = (ref_T0 + ref_T1)/2
    R_camera_to_cb = ref_R0

    points_checkerboard = np.dot(points_above_threshold, R_cb_to_torch) - np.dot(O_cb_to_torch, R_cb_to_torch)
    points_camera = np.dot(points_checkerboard, R_camera_to_cb) - np.dot(T_midpoint, R_camera_to_cb)
    
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(points_camera)
    gt_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    
    return gt_pcd


if __name__ == "__main__":
    
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # set the parser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
    
    ## Ground truth point cloud parameters
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples for the ground truth point cloud")
    parser.add_argument("--height_threshold", type=float, default=0.0015, help="Height threshold for the ground truth point cloud")
    
    ## Additional parameters
    parser.add_argument("--texture", action="store_true", help="Whether to do the textured version or not")
    parser.add_argument("--overlay", action="store_true", help="Whether to save overlayed images or not (for visualization)")
    parser.add_argument("--mask", action="store_true", help="Whether to save masked images or not")
    args = parser.parse_args()   
    
    directory_path = args.data_path
    objects_path = os.path.join(directory_path, 'objects')
    textured_objects_path = os.path.join(directory_path, 'textured_objects')
    objects = os.listdir(objects_path)
    objects.sort()
    
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
        for category in os.listdir(folders_path):
            category_path = os.path.join(folders_path, category)
            
            log = ''
            
            folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            for i,folder in enumerate(folders):
                
                print(f'Processing {directory}/{category}/{folder}..')
                obj_dir = os.path.join(category_path, folder)
                
                # loads the images
                # if the undistorted images already exist, loads them otherwise undistorts and saves them
                # returns a dictionary with the images
                images = preprocess_images(obj_dir, intrinsics, dist_coeffs)
                     
                best_pair = best_pair_selection(images)
                log += f'{best_pair[0]} {best_pair[1]}\n'
                
                # MESHES
                ################ Original mesh #################
                obj_filename = os.path.join(objects_path, objects[i])
                mesh = load_mesh(obj_filename, device, scaling_factor=0.001)
                

                ################ Textured mesh #################
                if args.texture:
                    textured_obj_filename = os.path.join(textured_objects_path, f'object0{i+1}', 'tinker.obj')
                    textured_mesh = load_mesh(textured_obj_filename, device, scaling_factor=0.001)
                
            
                ################ Ground Thruth Point Cloud #################
                gt_pointcloud = generate_ground_truth_pointcloud(mesh, 
                                                                [images[best_pair[0]], images[best_pair[1]]],
                                                                num_samples=args.num_samples,
                                                                height_threshold=args.height_threshold)
                o3d.io.write_point_cloud(obj_dir+"/gt_pointcloud.ply", gt_pointcloud)
                
                
                
                for image in images.values():
                    
                    image_name = image['name']
                    print("Processing image: ", image_name, "...")
                    
                    # render the image
                    renderer = set_renderer(mesh, image, intrinsics, device)
                    rendered_img = renderer(mesh)
                    
                    # save overlayed image
                    if args.overlay:
                        overlayed_path = obj_dir+"/overlayed"
                        os.makedirs(overlayed_path, exist_ok=True)
                        
                        overlay_images(image['undistorted_image'], 
                                    rendered_img[0, ..., :3].cpu().numpy()*255, 
                                    opacity=80, 
                                    save_path=overlayed_path+"/"+image_name+".jpg")
                    
                    
                    
                    # generate the depth map
                    fragments = renderer.rasterizer(mesh)
                    depth_map = fragments.zbuf.squeeze().cpu().numpy()
                    
                    depths_path = obj_dir+"/depths"
                    os.makedirs(depths_path, exist_ok=True)
                    save_pfm(depths_path+"/"+image_name+".pfm", depth_map)
  
                    # mask the image
                    if args.mask:
                        masked_path = obj_dir+"/masked"
                        os.makedirs(masked_path, exist_ok=True)
                        
                        mask = depth_map == -1
                        masked_img = image['undistorted_image'].copy()
                        masked_img[mask] = 0
                        cv2.imwrite(masked_path+"/"+image_name+".jpg", masked_img)


                    ############## Textured ################
                    if args.texture:
                        
                        textured_rendered_img = renderer(textured_mesh)
                
                        textured_rendered_image = textured_rendered_img[0, ..., :3].cpu().numpy()
                        rendered_gray = cv2.cvtColor(textured_rendered_image, cv2.COLOR_RGB2GRAY)

                        adjusted_image = (rendered_gray / np.max(rendered_gray)) * 255
                        adjusted_image = adjusted_image.astype(np.uint8)
                        mask = adjusted_image != 255
                        int_mask = mask.astype(np.uint8)

                        real_image = image['undistorted_image'][:,:,0]
                        opacity = 0.5

                        combined_pixels = adjusted_image * int_mask * opacity + real_image * int_mask * (1 - opacity)

                        combined_image = real_image.copy()
                        combined_image[mask] = combined_pixels[mask]
                        
                        # save the combined images
                        os.makedirs(obj_dir+"/textured", exist_ok=True)
                        cv2.imwrite(obj_dir+"/textured/"+image_name+".jpg", combined_image)
                        
                        if args.mask:
                            combined_image[~mask] = 0
                            os.makedirs(obj_dir+"/textured_masked", exist_ok=True)
                            cv2.imwrite(obj_dir+"/textured_masked/"+image_name+".jpg", combined_image)
                              
            with open(os.path.join(category_path, 'reference_cameras.txt'), 'w') as f:
                f.write(log)