Download and unzip the custom dataset from [this link](https://drive.google.com/file/d/1s7sRLkNbAUNJfIWSSCGxMmwya3Nix7VE/view?usp=sharing).

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
