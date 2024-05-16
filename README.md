# Multi-View Stereo Reconstruction Thesis Work

This repository contains the dataset and code for my thesis work on Multi-view Stereo (MVS) Depth Estimation and 3D Reconstruction. The project involves the use of three different MVS techniques: COLMAP, GBi-Net, and MVSformer, applied to a custom dataset captured with a cobot (collaborative robot).

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

Here's the modified "Usage" section including information about the preprocessing script and its parameters:

## Usage

### Dataset Preprocessing

To preprocess the dataset and prepare it for further analysis, you can use the provided `preprocessing.py` script located in the `src` folder. This script performs several preprocessing steps on the dataset, including loading the images, undistorting them, and saving the undistorted versions. Additionally, it saves the depth maps for each image and generates ground truth point clouds for each object.

#### Running the Preprocessing Script

You can run the preprocessing script with the following command:

```bash
python src/preprocessing.py --data_path=path_to_dataset [--additional_parameters]
```

#### Additional Parameters

- `--data_path`: Path to the directory containing the dataset (default: "data").

##### Ground Truth Point Cloud Parameters

- `--num_samples`: Number of samples for the ground truth point cloud (default: 100000).
- `--height_threshold`: Height threshold for the ground truth point cloud (default: 0.0015).

##### Additional Parameters

- `--texture`: Whether to generate the textured version or not.
- `--overlay`: Whether to save overlayed images for visualization.
- `--mask`: Whether to save masked images.


### Citation

If you use this dataset or the preprocessing script in your research, please cite the corresponding paper:

[Insert citation here]

### License

This dataset and the preprocessing script are provided under the [insert license here]. See the LICENSE file for details.

### Acknowledgments

[Insert acknowledgments or credits here, such as funding sources or collaborators.]

### Contact

For any questions or inquiries, please contact [insert your contact information].
```

This section provides clear instructions on how to use the preprocessing script along with explanations of the available parameters. Let me know if you need further adjustments!

- [List any acknowledgments or credits here, such as funding sources or collaborators.]

## Contact

For any questions or inquiries, please contact [insert your contact information].

