import unittest

import cv2
import numpy as np

from robo_manip_baselines.envs.real.RealEnvBase import RealEnvBase

SHOW_WIDTH = 480
SHOW_HEIGHT = 360

CAMERA_ID_FRONT = "145522067924"
CAMERA_ID_HAND = "153122070885"
TACTILE_ID_LEFT = "GelSight Mini R0B 2BNK-CE0U: Ge"
TACTILE_ID_RIGHT = "GelSight Mini R0B 2BG8-0H3X: Ge"
KEYBOARD_PATH_LEFT = "/dev/hidraw6"
KEYBOARD_PATH_RIGHT = "/dev/hidraw9"


class DummyRealEnv(RealEnvBase):
    def __init__(
        self,
        camera_ids=None,
        pointcloud_camera_ids=None,
        gelsight_ids=None,
        sanwa_keyboard_ids=None,
    ):
        super().__init__(
            robot_ip=None,
            camera_ids=camera_ids,
            gelsight_ids=gelsight_ids,
            pointcloud_camera_ids=pointcloud_camera_ids,
            sanwa_keyboard_ids=sanwa_keyboard_ids,
        )

        self.setup_realsense(camera_ids)

        self.setup_femtobolt(pointcloud_camera_ids)

        # The device ID may change if the USB cable is unplugged or connected to a different port.
        # Run `v4l2-ctl --list-devices` to check the current device ID for the camera.
        self.setup_gelsight(gelsight_ids)

        self.setup_sanwa_keyboard(sanwa_keyboard_ids)

    def _reset_robot(self, *args, **kwargs):
        pass

    def _set_action(self, *args, **kwargs):
        pass

    def _get_obs(self, *args, **kwargs):
        pass


class TestRealEnvBaseGetInfo(unittest.TestCase):
    def assert_env_info_valid(self, dummy_real_env):
        info = dummy_real_env._get_info()
        self.assert_camera_images_valid(dummy_real_env, info)
        self.assert_tactile_images_valid(dummy_real_env, info)
        self.assert_pointcloud_images_valid(dummy_real_env, info)
        self.assert_intensity_tactile_valid(dummy_real_env, info)

    def assert_camera_images_valid(self, dummy_real_env, info):
        for camera_name in dummy_real_env.camera_names:
            rgb_image = info["rgb_images"][camera_name]
            self.assertIsInstance(rgb_image, np.ndarray)
            self.assertEqual(rgb_image.dtype, np.uint8)
            self.assertEqual(len(rgb_image.shape), 3)
            self.assertEqual(rgb_image.shape[-1], 3)

            depth_image = info["depth_images"][camera_name]
            self.assertIsNotNone(depth_image)
            self.assertIsInstance(depth_image, np.ndarray)
            self.assertEqual(depth_image.dtype, np.float32)
            self.assertEqual(len(depth_image.shape), 2)

    def assert_pointcloud_images_valid(self, dummy_real_env, info):
        for pointcloud_camera_name in dummy_real_env.pointcloud_camera_names:
            rgb_image = info["rgb_images"][pointcloud_camera_name]
            self.assertIsInstance(rgb_image, np.ndarray)
            self.assertEqual(rgb_image.dtype, np.uint8)
            self.assertEqual(len(rgb_image.shape), 3)
            self.assertEqual(rgb_image.shape[-1], 3)

            depth_image = info["depth_images"][pointcloud_camera_name]
            self.assertIsNotNone(depth_image)
            self.assertIsInstance(depth_image, np.ndarray)
            self.assertEqual(depth_image.dtype, np.float32)
            self.assertEqual(len(depth_image.shape), 2)

            point_cloud = info["pointclouds"][pointcloud_camera_name]
            self.assertIsNotNone(point_cloud)
            self.assertIsInstance(point_cloud, np.ndarray)
            self.assertEqual(point_cloud.dtype, np.float32)
            self.assertEqual(len(point_cloud.shape), 2)

    def assert_tactile_images_valid(self, dummy_real_env, info):
        for rgb_tactile_name in dummy_real_env.rgb_tactile_names:
            rgb_image = info["rgb_images"][rgb_tactile_name]
            self.assertIsInstance(rgb_image, np.ndarray)
            self.assertEqual(rgb_image.dtype, np.uint8)
            self.assertEqual(len(rgb_image.shape), 3)
            self.assertEqual(rgb_image.shape[-1], 3)

            depth_image = info["depth_images"][rgb_tactile_name]
            self.assertIsNone(depth_image)

    def assert_intensity_tactile_valid(self, dummy_real_env, info):
        for intensity_tactile_name in dummy_real_env.intensity_tactile_names:
            intensity_tactile = info["intensity_tactile"][intensity_tactile_name]
            self.assertIsInstance(intensity_tactile, np.ndarray)
            self.assertEqual(intensity_tactile.dtype, np.uint8)
            self.assertEqual(len(intensity_tactile.shape), 2)
            self.assertEqual(intensity_tactile.shape[-1], 3)

    def show_image_loop(self, dummy_real_env):
        print(f"[{self.__class__.__name__}] Press q on image to exit.")
        try:
            while True:
                # get rgb image
                info = dummy_real_env._get_info()
                for camera_name in (
                    dummy_real_env.camera_names | dummy_real_env.rgb_tactile_names
                ):
                    rgb_image = info["rgb_images"][camera_name]
                    rgb_image = cv2.resize(rgb_image, (SHOW_WIDTH, SHOW_HEIGHT))
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow(camera_name, bgr_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_pointcloud_loop(self, dummy_real_env):
        import open3d as o3d

        print(f"[{self.__class__.__name__}] Press Ctrl+C to exit.")
        vis_list = []
        for i, camera_name in enumerate(dummy_real_env.pointcloud_camera_names):
            vis_list.append(o3d.visualization.Visualizer())
            vis_list[i].create_window()
        try:
            while True:
                info = dummy_real_env._get_info()
                for i, camera_name in enumerate(dummy_real_env.pointcloud_camera_names):
                    points = info["pointclouds"][camera_name]
                    if points is None:
                        continue
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    if points.shape[1] == 6:
                        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
                    vis_list[i].clear_geometries()
                    vis_list[i].add_geometry(pcd)
                    vis_list[i].poll_events()
                    vis_list[i].update_renderer()

        except Exception as e:
            print(e)

    def show_intensity_loop(self, dummy_real_env, vmin=-1.0, vmax=1.0):
        import matplotlib.pyplot as plt

        print(f"[{self.__class__.__name__}] Press q on plot to exit.")
        fig, axes = plt.subplots(2, 1)
        try:
            while True:
                info = dummy_real_env._get_info()
                for tactile_name, ax in zip(
                    dummy_real_env.intensity_tactile_names, axes
                ):
                    tactile_data = info["intensity_tactile"][tactile_name]
                    ax.clear()
                    ax.axis("off")
                    ax.imshow(
                        np.clip(tactile_data, vmin, vmax),
                        cmap="coolwarm",
                        interpolation="none",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_title(tactile_name)
                fig.canvas.draw()
                cv2.imshow(
                    "TestTactileIntensity",
                    cv2.cvtColor(
                        np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR
                    ),
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_camera1(self):
        dummy_real_env = DummyRealEnv(
            camera_ids={"front": CAMERA_ID_FRONT},
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_camera2(self):
        dummy_real_env = DummyRealEnv(
            camera_ids={"front": CAMERA_ID_FRONT, "hand": CAMERA_ID_HAND},
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_pointcloud_camera(self):
        dummy_real_env = DummyRealEnv(pointcloud_camera_ids={"femtobolt": 0})
        while True:
            info = dummy_real_env._get_info()
            if info["pointclouds"]:
                break
        self.assert_env_info_valid(dummy_real_env)
        self.show_pointcloud_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_rgb_tactile1(self):
        dummy_real_env = DummyRealEnv(
            gelsight_ids={
                "tactile_left": TACTILE_ID_LEFT,
            },
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_rgb_tactile2(self):
        dummy_real_env = DummyRealEnv(
            gelsight_ids={
                "tactile_right": TACTILE_ID_RIGHT,
            },
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_rgb_tactile3(self):
        dummy_real_env = DummyRealEnv(
            gelsight_ids={
                "tactile_left": TACTILE_ID_LEFT,
                "tactile_right": TACTILE_ID_RIGHT,
            },
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_camera_rgb_tactile1(self):
        dummy_real_env = DummyRealEnv(
            camera_ids={"front": CAMERA_ID_FRONT},
            gelsight_ids={
                "tactile_right": TACTILE_ID_RIGHT,
            },
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_camera_rgb_tactile2(self):
        dummy_real_env = DummyRealEnv(
            camera_ids={"front": CAMERA_ID_FRONT},
            gelsight_ids={
                "tactile_left": TACTILE_ID_LEFT,
                "tactile_right": TACTILE_ID_RIGHT,
            },
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_intensity_tactile(self):
        dummy_real_env = DummyRealEnv(
            sanwa_keyboard_ids={
                "tactile_left": KEYBOARD_PATH_LEFT,
                "tactile_right": KEYBOARD_PATH_RIGHT,
            },
        )
        while True:
            info = dummy_real_env._get_info()
            if info["intensity_tactile"]:
                break
        self.assert_env_info_valid(dummy_real_env)
        self.show_intensity_loop(dummy_real_env)


if __name__ == "__main__":
    unittest.main()
