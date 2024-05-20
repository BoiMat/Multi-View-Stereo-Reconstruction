# Multi-View Stereo Reconstruction Thesis Work

This repository contains the dataset and code for my thesis work on Multi-view Stereo (MVS) Depth Estimation and 3D Reconstruction. The project involves the use of three different MVS techniques: COLMAP, GBi-Net, and MVSformer, applied to a custom dataset captured with a URX10e collaborative robot.

## Dataset Structure

The dataset is structured as follows:

- **1movement_cb/**: Images captured with the cobot in a single movement with calibration board.
  - **artificial/**: Images captured with artificial light.
  - **natural/**: Images captured with natural light.
    
- **2movements_cb/**: Similar structure to `1movement_cb/`, but with two movements.
- **1movement_nocb/**: Similar structure to `1movement_cb/`, but without the checkerboard underneath the objects.

Each subfolder (namely **scan01/**, **scan02/**, etc.) contains images of the six objects captured from different viewpoints, along with pose information in `pose.yml`.

- **objects/**: 3D meshes of the objects used in the dataset.

- **textured_objects/**: Textured versions of the 3D objects with added artificial lines pattern.

- **calibrated_camera.yml**: Contains camera calibration specifics including intrinsic parameters and distortion coefficients.

## Usage

### Dataset Preprocessing

To preprocess the dataset and prepare it for further analysis, you can use the provided `preprocessing.py` script located in the `src` folder. This script performs several preprocessing steps on the dataset, including loading the images, undistorting them, and saving the undistorted versions. Additionally, it saves the depth maps for each image and generates ground truth point clouds for each object.

#### Running the Preprocessing Script

You can run the preprocessing script with the following command:

```bash
python src/preprocessing.py --data_path=path_to_dataset [--additional_parameters]
```

- `--data_path`: Path to the directory containing the dataset (default: "data").

#### Additional Parameters
##### Ground Truth Point Cloud Parameters

- `--num_samples`: Number of samples for the ground truth point cloud (default: 100000).
- `--height_threshold`: Height threshold for the ground truth point cloud (default: 0.0015).

##### Additional Parameters

- `--texture`: Whether to generate the textured version or not.
- `--overlay`: When this option is enabled, the script will generate images where the original image is overlaid with the 3D mesh, highlighting it in blue.
- `--mask`: When this option is enabled, the script generates masked images where the background is completely black, and only the portion where the object is present remains visible.

### COLMAP Reconstruction

Before using the deep models, it's necessary to reconstruct the scene using COLMAP. COLMAP is a popular Structure-from-Motion (SfM) and Multi-View Stereo (MVS) software. You can download and install it from the [official website](https://colmap.github.io/).

#### Setup for COLMAP Reconstruction

1. **Prepare for Reconstruction**: Run the `custom2colmap.py` script located in the `src` folder. This script sets up a folder structure for COLMAP reconstruction. It takes the data path and the output path as parser parameters.

   ```bash
   python src/custom2colmap.py --data_path=path_to_dataset --output_path=output_folder
   ```

#### COLMAP Reconstruction Process

COLMAP reconstruction consists of two main steps: a fast, sparse reconstruction, and a subsequent slow, deep reconstruction.

1. **Fast, Sparse Reconstruction**: This step generates a quick, sparse reconstruction of the scene, which is sufficient for using the deep models.

2. **Slow, Dense Reconstruction**: After the fast reconstruction, a more detailed, deep reconstruction can be performed. This step provides a more accurate 3D model of the scene.

#### Running COLMAP Reconstruction

- **Windows**: For Windows users, a `.bat` script called `run_colmap.bat` is provided to facilitate running COLMAP using the pre-built binaries. Place the `run_colmap.bat` file in the same folder as the COLMAP binaries. Then, simply execute the script and follow the example instructions.

- **Linux/Mac**: follow the official COLMAP documentation to install and run COLMAP on Linux or Mac systems.



### License

This dataset and the preprocessing script are provided under the [insert license here]. See the LICENSE file for details.
```
