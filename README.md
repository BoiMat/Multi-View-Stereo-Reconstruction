# Multi-View Stereo Reconstruction Thesis Work

This repository contains the dataset and code for my thesis work on Multi-view Stereo (MVS) Depth Estimation and 3D Reconstruction. The project involves the use of three different MVS techniques: COLMAP, GBi-Net, and MVSformer, applied to a custom dataset captured with a UR10e collaborative robot.

## Requirements

This project involves running tests with three state-of-the-art models for multi-view stereo depth estimation and 3D reconstruction. We used a traditional method (COLMAP), a CNN-based method (gbinet), and a vision transformer-based method (MVSformer) to ensure a variety of approaches.

- To avoid conflicts, it is recommended to use **Python 3.8** for this project
- **PyTorch3D** can be installed by following the instructions provided in the [official PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
- **COLMAP** is a popular Structure-from-Motion (SfM) and Multi-View Stereo (MVS) software. You can download and install it from the [official website](https://colmap.github.io/).
- **GBi-Net** is a CNN-based multi-view stereo method. The code and installation instructions are on the [author's GitHub page](https://github.com/MiZhenxing/GBi-Net).
- **MVSformer** is a vision transformer-based multi-view stereo method. The code and installation instructions are on the [author's GitHub page](https://github.com/MVSformer/MVSFormer).

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

Before using the deep models, it's necessary to run COLMAP sparse reconstruction.

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


### GBi-Net and MVSFormer Reconstruction

In the `src` folder, you will find two subfolders: `gbinet` and `mvsformer`. These folders contain additional files and scripts that should be added after completing the installation as instructed on the original GitHub pages of GBi-Net and MVSFormer. These additional files include custom configurations and any modifications necessary to run the models with our custom dataset. Follow the installation instructions on the respective GitHub pages first, and then incorporate the contents of these subfolders to complete the setup.
