import numpy as np
import cv2
import os
from PIL import Image

def load_intrinsics(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    intrinsics = np.array(fs.getNode("CameraMatrix").mat())
    distortion_coeffs = np.array(fs.getNode("DistCoeffs").mat())
    fs.release()
    return intrinsics, distortion_coeffs.reshape(-1)



def load_camera_poses(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_poses = np.array(fs.getNode("CameraInCheckerboard").mat())
    fs.release()
    return camera_poses



def pose_to_transformation_matrix(pose):
    # Extract pose components
    T = np.array(pose[:3])
    # Convert Rodrigues vector to rotation matrix
    rvec = np.array(pose[3:])
    R, _ = cv2.Rodrigues(rvec)

    return R, T


def undistort_images(images, intrinsics, distortion_coeffs, undistorted_path=None):
    if undistorted_path:
        os.makedirs(undistorted_path, exist_ok=True)
        
    for i in images:
        undistorted_img = cv2.undistort(images[i]['image'], intrinsics, distortion_coeffs)
        images[i]['undistorted_image'] = undistorted_img
        
        if undistorted_path:
            cv2.imwrite(os.path.join(undistorted_path, images[i]['name'] + '.jpg'), undistorted_img)
            

def load_images(image_path, camera_poses, undistorted_path=None, format='jpg'):
    images = {}
    image_names = [f for f in os.listdir(image_path) if f.endswith(format)]
    image_names.sort()
        
    for i,file in enumerate(image_names):
        img = cv2.imread(os.path.join(image_path, file))
        
        # rename the image as 00000000.jpg, 00000001.jpg, ...
        new_file = '{:08d}.jpg'.format(i)
        os.rename(os.path.join(image_path, file), os.path.join(image_path, new_file))
        
        img_array = np.array(img)
        R, T = pose_to_transformation_matrix(camera_poses[i])

        images[i] = {'name': new_file.split('.')[0],
                    'image':img_array, 
                    'R': R, 
                    'T': T}
        
        if undistorted_path:
            undistorted_img = cv2.imread(os.path.join(undistorted_path, file))
            images[i]['undistorted_image'] = undistorted_img
        
    return images


def find_feature_matches(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches


def computer_scores(images):
    score = np.zeros((len(images), len(images)))
    queue = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            queue.append((i, j))

    def calc_score(inputs):
        i, j = inputs
        img1 = images[i]['image']
        img2 = images[j]['image']
        kp1, kp2, matches = find_feature_matches(img1, img2)
        return [i, j, len(matches)]
    
    for q in queue:
        i, j, s = calc_score(q)
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:10]])
    
    return view_sel


def compute_relative_pose_to_midpoint(R0, T0, R1, T1, Ri, Ti):
    """
    Computes the pose of camera 'i' relative to the midpoint between two reference cameras.

    Parameters:
    - R0, T0: Rotation matrix and translation vector of the first reference camera.
    - R1, T1: Rotation matrix and translation vector of the second reference camera.
    - Ri, Ti: Rotation matrix and translation vector of the ith camera.

    Returns:
    - R_relative, T_relative: The relative rotation and translation of camera 'i'.
    """
    # Calculate the midpoint between T0 and T1
    T_midpoint = (T0 + T1) / 2

    # For simplicity, let's assume the rotation of the first camera represents the orientation well enough.
    R_midpoint = R0

    # Compute the relative pose of the ith camera to the midpoint
    R_relative = np.dot(np.linalg.inv(R_midpoint), Ri)
    T_relative = np.dot(np.linalg.inv(R_midpoint), Ti - T_midpoint)

    return R_relative, T_relative


def save_colmap_dataset(images, current_folder, intrinsics, dist_coeffs):
    
    colmap_dir = os.path.join('colmap', current_folder)
    os.makedirs(colmap_dir, exist_ok=True)
    
    colmap_dir_sparse = colmap_dir + '/sparse'
    os.makedirs(colmap_dir_sparse, exist_ok=True)
    
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: 0\n".format(len(images)))
    with open(colmap_dir_sparse + '/images.txt', 'w') as f:
        f.write(HEADER)
        for i,image in images.items():
            f.write("{} {} {} 1 {}.jpg\n\n".format(i+1, 
                                                ' '.join(map(str, image['colmap'][0])),
                                                ' '.join(map(str, image['colmap'][1])),
                                                image['name']))
            
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: 1\n")
    with open(colmap_dir_sparse + '/cameras.txt', 'w') as fid:
        fid.write(HEADER)
        cam_intrinsic = [intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]]
        fid.write("1 OPENCV {} {} {} {} {} {} {} {} {} {}\n".format(images[0]['undistorted_image'].shape[1],
                                                                    images[0]['undistorted_image'].shape[0],
                                                                    *cam_intrinsic, *dist_coeffs[:4]))
        
    with open(colmap_dir_sparse+'/points3D.txt', 'w') as f:
        f.write("")
        
        


def checkerboard_to_view(R_camera_to_cb, T_camera_to_cb, O_cb_to_torch):

    R_cb_to_torch = np.array([[0,1,0],
                            [1,0,0],
                            [0,0,-1]])

    R_camera_to_torch = R_cb_to_torch @ R_camera_to_cb
    T_camera_to_torch = R_cb_to_torch @ T_camera_to_cb + O_cb_to_torch

    R_adjusted = R_camera_to_torch @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T_final = - R_adjusted.T @ T_camera_to_torch
    
    return R_adjusted, T_final

def overlay_images(image, overlay, opacity=80, save_path=None):
    # Open the background and overlay images
    background = Image.fromarray(image.astype(np.uint8))
    overlay = Image.fromarray(overlay.astype(np.uint8))

    # Resize overlay to match the background size
    overlay = overlay.resize(background.size, Image.LANCZOS)
    # Convert overlay image to RGBA (if not already in that mode)
    overlay = overlay.convert("RGBA")

    # Adjust the overlay opacity
    overlay_with_opacity = overlay.copy()
    for x in range(overlay.width):
        for y in range(overlay.height):
            current_color = overlay.getpixel((x, y))
            overlay_with_opacity.putpixel((x, y), current_color[:3] + (opacity,))

    # Composite the images
    blended = Image.alpha_composite(background.convert("RGBA"), overlay_with_opacity)

    # Convert back to RGB and show/save the result
    blended = blended.convert("RGB")
    if save_path:
        blended.save(save_path)


def save_pfm(file, image, scale=1):
    """
    Save a Numpy array to a PFM file.
    """
    with open(file, 'wb') as f:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)  # PFM files store rows in inverse order

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # grayscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        f.write(b'PF\n' if color else b'Pf\n')
        f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and np.little_endian:
            scale = -scale

        f.write(b'%f\n' % scale)

        image.tofile(f)