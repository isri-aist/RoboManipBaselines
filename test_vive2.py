#!/usr/bin/env python3
import numpy as np
import openvr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class ViveTrackerVisualizer:
    def __init__(self, y_offset=4.0, x_offset=2.0):
        self.vr_system = openvr.init(openvr.VRApplication_Other)

        # トラッカーのシリアル番号 → 名前
        self.device_sn_to_pose_name_map = {
            "LHR-8C30BD01": "root",
            "LHR-1FB29FC6": "target",
        }

        self.positions = {name: np.zeros(3) for name in self.device_sn_to_pose_name_map.values()}
        self.orientations = {name: np.eye(3) for name in self.device_sn_to_pose_name_map.values()}

        self.y_offset = y_offset
        self.x_offset = x_offset

        # matplotlib setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatter_root = self.ax.scatter([], [], [], c='blue', s=80, label='root')
        self.scatter_target = self.ax.scatter([], [], [], c='red', s=80, label='target')
        (self.line,) = self.ax.plot([], [], [], color='gray', linestyle='--')

        # 各トラッカーの3軸（X, Y, Z）を保持（quiver再利用）
        self.quiver_length = 0.2
        self.quivers = {}
        for name in ["root", "target"]:
            self.quivers[name] = {
                "x": self.ax.quiver(0, 0, 0, 0, 0, 0, color='r', length=self.quiver_length, normalize=True),
                "y": self.ax.quiver(0, 0, 0, 0, 0, 0, color='g', length=self.quiver_length, normalize=True),
                "z": self.ax.quiver(0, 0, 0, 0, 0, 0, color='b', length=self.quiver_length, normalize=True),
            }

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.legend()
        self.ax.set_title("Vive Tracker 3D Visualization (Optimized Quivers)")

    def __del__(self):
        openvr.shutdown()

    def update_vr_data(self):
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        for device_idx in range(openvr.k_unMaxTrackedDeviceCount):
            if not poses[device_idx].bDeviceIsConnected or not poses[device_idx].bPoseIsValid:
                continue

            device_sn = self.vr_system.getStringTrackedDeviceProperty(
                device_idx, openvr.Prop_SerialNumber_String
            )

            if device_sn not in self.device_sn_to_pose_name_map:
                continue

            pose_name = self.device_sn_to_pose_name_map[device_sn]
            pose_mat = np.zeros((4, 4))
            pose_mat[0:3, 0:4] = poses[device_idx].mDeviceToAbsoluteTracking.m
            pose_mat[3, 3] = 1.0

            pos = pose_mat[0:3, 3]
            R = pose_mat[0:3, 0:3]

            # 表示用オフセット
            pos[0] += self.x_offset
            pos[1] += self.y_offset

            self.positions[pose_name] = pos
            self.orientations[pose_name] = R

    def update_quiver(self, quiver_obj, pos, dir_vec):
        """
        既存quiverを再利用してベクトルの始点と方向を更新
        """
        # quiverはLine3DCollectionを1本保持しているので、そのベクトルを直接上書きする
        quiver_obj.set_segments([np.array([pos, pos + dir_vec])])

    def update_plot(self, frame):
        self.update_vr_data()

        # 点更新
        self.scatter_root._offsets3d = (
            [self.positions["root"][0]],
            [self.positions["root"][1]],
            [self.positions["root"][2]],
        )
        self.scatter_target._offsets3d = (
            [self.positions["target"][0]],
            [self.positions["target"][1]],
            [self.positions["target"][2]],
        )

        # 線更新
        self.line.set_data(
            [self.positions["root"][0], self.positions["target"][0]],
            [self.positions["root"][1], self.positions["target"][1]],
        )
        self.line.set_3d_properties(
            [self.positions["root"][2], self.positions["target"][2]]
        )

        # フレーム軸の更新（再生成しない）
        for name in ["root", "target"]:
            pos = self.positions[name]
            R = self.orientations[name]

            # X, Y, Z 軸を更新
            axes = ["x", "y", "z"]
            colors = ["r", "g", "b"]
            for i, axis in enumerate(axes):
                dir_vec = R[:, i] * self.quiver_length
                self.update_quiver(self.quivers[name][axis], pos, dir_vec)

        return [self.scatter_root, self.scatter_target, self.line]

    def run(self):
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            blit=False,
            cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    visualizer = ViveTrackerVisualizer(y_offset=4.0, x_offset=-2.0)
    visualizer.run()
