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

## Usage

To use this dataset, simply clone or download this repository. You can then access the images, poses, and object meshes for use in your research or experiments.

## Citation

If you use this dataset in your research, please cite the corresponding paper:

[Insert citation here]

## License

This dataset is provided under the [insert license here]. See the LICENSE file for details.

## Acknowledgments

- [List any acknowledgments or credits here, such as funding sources or collaborators.]

## Contact

For any questions or inquiries, please contact [insert your contact information].

