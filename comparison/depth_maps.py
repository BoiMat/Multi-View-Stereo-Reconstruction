import numpy as np
import os
import re
from argparse import ArgumentParser

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        while num_delimiter < 3:
            if fid.read(1) == b"&":
                num_delimiter += 1
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_pfm(filename):
    with open(filename, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()
        color = header == 'PF'
        if not (color or header == 'Pf'):
            raise Exception('Not a PFM file.')

        width, height = map(int, file.readline().decode('utf-8').split())
        scale = float(file.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        if scale < 0:
            scale = -scale

        data = np.fromfile(file, endian + 'f').reshape((height, width, 3) if color else (height, width))
        return np.flipud(data), scale

def compute_absrel(pred_depth, gt_depth):
    valid_mask = (gt_depth > 0)
    return np.mean(np.abs(pred_depth[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask])

def compute_sqrel(pred_depth, gt_depth):
    valid_mask = (gt_depth > 0)
    return np.mean(np.square(pred_depth[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask])

def compute_rmse(pred_depth, gt_depth):
    valid_mask = (gt_depth > 0)
    return np.sqrt(np.mean(np.square(pred_depth[valid_mask] - gt_depth[valid_mask])))

def calculate_depth_error_percentage(predicted_depth, ground_truth_depth, thresholds_mm):

    abs_diff = np.abs(predicted_depth - ground_truth_depth)
    error_percentages = {}

    for threshold in thresholds_mm:
        error_pixels = np.sum(abs_diff > threshold)
        total_pixels = predicted_depth.size
        error_percentage = (error_pixels / total_pixels) * 100
        error_percentages[f"Threshold {threshold}mm"] = error_percentage
    
    return error_percentages


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='data/')
    parser.add_argument('--colmap_path', type=str, default='colmap/')
    parser.add_argument('--gbinet_path', type=str, default='GBi-Net/exps/')
    parser.add_argument('--mvsf_path', type=str, default='MVSFormer/exps/')
    parser.add_argument('--results_path', type=str, default='results/')
    args = parser.parse_args()
    
    with open('folders_list.txt', 'r') as file:
        folders_list = [line.strip() for line in file]

    for folder in folders_list:
        directories = [d for d in os.listdir(os.path.join(args.gt_path, folder)) if os.path.isdir(os.path.join(args.gt_path, folder, d))]
        everything = {dir: {} for dir in directories}
        
        for dir in directories:
            gt_path = os.path.join(args.gt_path, folder, dir, 'depths')
            colmap_path = os.path.join(args.colmap_path, folder, dir, 'dense', 'stereo', 'depth_maps')
            gbinet_path = os.path.join(args.gbinet_path, folder, 'test_output', 'depth_8', dir)
            mvsf_path = os.path.join(args.mvsf_path, folder, dir, 'depth_est')

            gt_files = [os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.endswith('pfm')]
            colmap_files = sorted([os.path.join(colmap_path, f) for f in os.listdir(colmap_path) if f.endswith('geometric.bin')])
            gbinet_files = sorted([os.path.join(gbinet_path, f) for f in os.listdir(gbinet_path) if f.endswith('init.pfm')])
            mvsf_files = sorted([os.path.join(mvsf_path, f) for f in os.listdir(mvsf_path) if 'filtered' not in f and f.endswith('pfm')])

            everything[dir]['gt'] = [read_pfm(fp)[0] for fp in gt_files]
            everything[dir]['colmap'] = [read_array(fp) for fp in colmap_files]
            everything[dir]['gbinet'] = [read_pfm(fp)[0] for fp in gbinet_files]
            everything[dir]['mvsf'] = [read_pfm(fp)[0] for fp in mvsf_files]

        error_metrics = {'colmap': [], 'gbinet': [], 'mvsf': []}
        abs_rel = {'colmap': [], 'gbinet': [], 'mvsf': []}
        sqrel_error = {'colmap': [], 'gbinet': [], 'mvsf': []}
        rmse_error = {'colmap': [], 'gbinet': [], 'mvsf': []}

        for i in range(len(everything['scan01']['gt'])):
            dm_gt = everything['scan01']['gt'][i]
            dm_colmap = everything['scan01']['colmap'][i]
            dm_gbinet = everything['scan01']['gbinet'][i]
            dm_mvsf = everything['scan01']['mvsf'][i]

            min_height, min_width = np.min([dm_gt.shape, dm_colmap.shape, dm_gbinet.shape, dm_mvsf.shape], axis=0)

            resize_and_crop = lambda dm: dm[(dm.shape[0] - min_height) // 2:(dm.shape[0] + min_height) // 2,
                                            (dm.shape[1] - min_width) // 2:(dm.shape[1] + min_width) // 2]
            dm_gt_resized = resize_and_crop(dm_gt)
            dm_colmap_resized = resize_and_crop(dm_colmap)
            dm_gbinet_resized = resize_and_crop(dm_gbinet)
            dm_mvsf_resized = resize_and_crop(dm_mvsf)

            mask = dm_gt_resized != -1
            dm_gt_masked = dm_gt_resized[mask]
            dm_colmap_masked = dm_colmap_resized[mask]
            dm_gbinet_masked = dm_gbinet_resized[mask]
            dm_mvsf_masked = dm_mvsf_resized[mask]

            thresholds_mm = [1, 2, 3, 4]

            error_metrics['colmap'].append(calculate_depth_error_percentage(dm_colmap_masked * 1000, dm_gt_masked * 1000, thresholds_mm))
            error_metrics['gbinet'].append(calculate_depth_error_percentage(dm_gbinet_masked * 1000, dm_gt_masked * 1000, thresholds_mm))
            error_metrics['mvsf'].append(calculate_depth_error_percentage(dm_mvsf_masked * 1000, dm_gt_masked * 1000, thresholds_mm))

            abs_rel['colmap'].append(compute_absrel(dm_colmap_masked, dm_gt_masked))
            abs_rel['gbinet'].append(compute_absrel(dm_gbinet_masked, dm_gt_masked))
            abs_rel['mvsf'].append(compute_absrel(dm_mvsf_masked, dm_gt_masked))

            sqrel_error['colmap'].append(compute_sqrel(dm_colmap_masked, dm_gt_masked))
            sqrel_error['gbinet'].append(compute_sqrel(dm_gbinet_masked, dm_gt_masked))
            sqrel_error['mvsf'].append(compute_sqrel(dm_mvsf_masked, dm_gt_masked))

            rmse_error['colmap'].append(compute_rmse(dm_colmap_masked, dm_gt_masked))
            rmse_error['gbinet'].append(compute_rmse(dm_gbinet_masked, dm_gt_masked))
            rmse_error['mvsf'].append(compute_rmse(dm_mvsf_masked, dm_gt_masked))

        mean_percentage_errors = {key: np.mean([list(d.values()) for d in value], axis=0) for key, value in error_metrics.items()}

        file_path = folder.replace('/', '_') + '.txt'
        with open(file_path, 'w') as file:
            file.write("Absolute Relative Error:\n")
            for method, errors in abs_rel.items():
                file.write(f"{method}: {np.mean(errors)}\n")

            file.write("\nSquared Relative Error:\n")
            for method, errors in sqrel_error.items():
                file.write(f"{method}: {np.mean(errors)}\n")

            file.write("\nRMSE Error:\n")
            for method, errors in rmse_error.items():
                file.write(f"{method}: {np.mean(errors)}\n")

            file.write("\nError Percentages:\n")