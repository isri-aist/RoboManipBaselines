import numpy as np

from robo_manip_baselines.common import DataKey, RmbData


class TrainPointCloudMixin:
    "This mixin is intended to be used in classes that inherit from TrainBase."

    def setup_pointcloud_info(self):
        pc_key = DataKey.get_pointcloud_key(self.args.camera_names[0])
        num_points = None
        image_size = None
        min_bound = None
        max_bound = None
        rpy_angle = None

        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                num_points_new = rmb_data[pc_key].shape[1]
                if num_points is None:
                    num_points = num_points_new
                elif num_points != num_points_new:
                    raise ValueError(
                        f"[{self.__class__.__name__}] num_points is inconsistent in dataset: {num_points} != {num_points_new}"
                    )

                image_size_new = rmb_data.attrs[pc_key + "_image_size"]
                if image_size is None:
                    image_size = image_size_new
                elif not np.array_equal(image_size, image_size_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] image_size is inconsistent in dataset: {image_size} != {image_size_new}"
                    )

                min_bound_new = rmb_data.attrs[pc_key + "_min_bound"]
                if min_bound is None:
                    min_bound = min_bound_new
                elif not np.allclose(min_bound, min_bound_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] min_bound is inconsistent in dataset: {min_bound} != {min_bound_new}"
                    )

                max_bound_new = rmb_data.attrs[pc_key + "_max_bound"]
                if max_bound is None:
                    max_bound = max_bound_new
                elif not np.allclose(max_bound, max_bound_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] max_bound is inconsistent in dataset: {max_bound} != {max_bound_new}"
                    )

                rpy_angle_new = rmb_data.attrs[pc_key + "_rpy_angle"]
                if rpy_angle is None:
                    rpy_angle = rpy_angle_new
                elif not np.allclose(rpy_angle, rpy_angle_new):
                    raise ValueError(
                        f"[{self.__class__.__name__}] rpy_angle is inconsistent in dataset: {rpy_angle} != {rpy_angle_new}"
                    )

        return num_points, image_size, min_bound, max_bound, rpy_angle

    def set_pointcloud_stats(self):
        pc_key = DataKey.get_pointcloud_key(self.args.camera_names[0])
        all_pointcloud = []
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                pointcloud = rmb_data[pc_key][:: self.args.skip]
                all_pointcloud.append(pointcloud.reshape(-1, pointcloud.shape[-1]))
        all_pointcloud = np.concatenate(all_pointcloud, dtype=np.float64)

        self.model_meta_info["pointcloud"] = self.calc_stats_from_seq(all_pointcloud)
