import argparse

import cv2
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_depth_image_to_pointcloud,
    euler_to_rotation_matrix,
    find_rmb_files,
)
from robo_manip_baselines.common.utils.Vision3dUtils import (
    crop_pointcloud_bb,
    downsample_pointcloud_fps,
    rotate_pointcloud,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="front",
        help="name of the camera to which the point cloud will be added",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[84, 84],
        help="image size (width, height) to be resized",
    )
    parser.add_argument(
        "--min_bound",
        type=float,
        nargs=3,
        default=[-0.4, -0.4, -0.4],
        help="min bounds of the bounding box for cropping",
    )
    parser.add_argument(
        "--max_bound",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="max bounds of the bounding box for cropping",
    )
    parser.add_argument(
        "--rpy_angle",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="rotation of the bounding box for cropping",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=512,
        help="number of points in a point cloud",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class AddPointCloudToRmbData:
    def __init__(
        self,
        path,
        camera_name,
        image_size,
        min_bound=None,
        max_bound=None,
        rpy_angle=None,
        num_points=512,
        overwrite=False,
    ):
        self.path = path
        self.camera_name = camera_name
        self.image_size = image_size
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.rpy_angle = rpy_angle
        self.num_points = num_points
        self.overwrite = overwrite

    def run(self):
        pc_key = DataKey.get_pointcloud_key(self.camera_name)
        print(
            f"[{self.__class__.__name__}] Add pointcloud '{pc_key}' generated from RGB and depth images."
        )
        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                if pc_key in rmb_data.keys():
                    if self.overwrite:
                        del rmb_data.h5file[pc_key]
                    else:
                        raise ValueError(
                            f"[{self.__class__.__name__}] Pointcloud already exists: {rmb_path} (use --overwrite to replace)"
                        )

                if pc_key + "_raw" in rmb_data.keys():
                    self.raw_pointcloud_exist = True
                    self.image_size = [-1, -1]
                else:
                    self.raw_pointcloud_exist = False

                pointclouds = self.get_pointclouds(rmb_data)

                rmb_data.h5file[pc_key] = pointclouds
                rmb_data.attrs[pc_key + "_image_size"] = self.image_size
                rmb_data.attrs[pc_key + "_min_bound"] = self.min_bound
                rmb_data.attrs[pc_key + "_max_bound"] = self.max_bound
                rmb_data.attrs[pc_key + "_rpy_angle"] = self.rpy_angle

    def get_pointclouds(self, rmb_data):
        pointclouds = []

        if self.raw_pointcloud_exist:
            pc_key = DataKey.get_pointcloud_key(self.camera_name)
            raw_pointclouds = rmb_data[pc_key + "_raw"][:]
            for i in range(len(raw_pointclouds)):
                # Get pointcloud
                pointcloud = raw_pointclouds[i]

                # Crop and downsample pointcloud
                rotmat = euler_to_rotation_matrix(self.rpy_angle)
                pointcloud = rotate_pointcloud(pointcloud, rotmat)
                pointcloud = crop_pointcloud_bb(
                    pointcloud, self.min_bound, self.max_bound
                )
                pointcloud = downsample_pointcloud_fps(pointcloud, self.num_points)

                pointclouds.append(pointcloud)
        else:
            # Load images
            rgb_image_seq = rmb_data[DataKey.get_rgb_image_key(self.camera_name)][:]
            depth_image_seq = rmb_data[DataKey.get_depth_image_key(self.camera_name)][:]
            fovy = rmb_data.attrs[
                DataKey.get_depth_image_key(self.camera_name) + "_fovy"
            ]

            # Resize images
            rgb_image_seq = np.array(
                [cv2.resize(image, self.image_size) for image in rgb_image_seq]
            )
            depth_image_seq = np.array(
                [cv2.resize(image, self.image_size) for image in depth_image_seq]
            )

            # Generate pointcloud
            for rgb_image, depth_image in zip(rgb_image_seq, depth_image_seq):
                # Convert to pointcloud
                pointcloud = np.concat(
                    convert_depth_image_to_pointcloud(depth_image, fovy, rgb_image),
                    axis=1,
                )

                # Crop and downsample pointcloud
                rotmat = euler_to_rotation_matrix(self.rpy_angle)
                pointcloud = rotate_pointcloud(pointcloud, rotmat)
                pointcloud = crop_pointcloud_bb(
                    pointcloud, self.min_bound, self.max_bound
                )
                pointcloud = downsample_pointcloud_fps(pointcloud, self.num_points)

                pointclouds.append(pointcloud)

        return np.array(pointclouds)


if __name__ == "__main__":
    add_point_cloud = AddPointCloudToRmbData(**vars(parse_argument()))
    add_point_cloud.run()
