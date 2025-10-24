#!/usr/bin/env python3
"""
Vispy-based real-time Vive tracker visualizer.
- GPU/OpenGL を使って高速に描画します。
- root / target の位置（点）と、それぞれのローカル軸（X/Y/Z: r/g/b）を線で表示。
- root-target 間はラインで結びます。
- 描画は Timer（約60Hz）で更新。
- x_offset, y_offset を引数で指定可。
"""
import sys
import numpy as np
import openvr
from vispy import app, scene, visuals, color

class ViveVisualizerVispy:
    def __init__(self, x_offset=2.0, y_offset=4.0, quiver_len=0.2, update_hz=60.0):
        # OpenVR init
        self.vr_system = openvr.init(openvr.VRApplication_Other)

        # シリアル→名称マッピング（必要に応じて変更）
        self.device_sn_to_pose_name_map = {
            "LHR-8C30BD01": "root",
            "LHR-1FB29FC6": "target",
        }
        # 初期データ
        self.positions = {name: np.zeros(3) for name in self.device_sn_to_pose_name_map.values()}
        self.orientations = {name: np.eye(3) for name in self.device_sn_to_pose_name_map.values()}

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.quiver_len = quiver_len

        # vispy Canvas + Scene
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(1000, 700), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=60, distance=3.5, up='+z')  # 回転可能カメラ

        # 軸グリッド（見やすくする）
        grid = visuals.GridLines(parent=self.view.scene)

        # Markers (points)
        self.marker_root = visuals.Markers(parent=self.view.scene)
        self.marker_target = visuals.Markers(parent=self.view.scene)

        # Connection line
        self.conn_line = visuals.Line(parent=self.view.scene, width=2.0, method='gl', connect='strip')

        # For each tracker: create 3 Line visuals for X/Y/Z axis
        self.axis_lines = {}
        for name in ["root", "target"]:
            self.axis_lines[name] = {
                "x": visuals.Line(parent=self.view.scene, width=3.0, color=(1,0,0,1), method='gl', connect='segments'),
                "y": visuals.Line(parent=self.view.scene, width=3.0, color=(0,1,0,1), method='gl', connect='segments'),
                "z": visuals.Line(parent=self.view.scene, width=3.0, color=(0,0,1,1), method='gl', connect='segments'),
            }

        # optional: small sphere meshes at arrow tips to visualize direction better
        self.tips = {name: [visuals.Markers(parent=self.view.scene) for _ in range(3)] for name in ["root", "target"]}

        # labels (text)
        self.text_root = visuals.Text("root", parent=self.view.scene, color='black', font_size=12, anchor_x='left')
        self.text_target = visuals.Text("target", parent=self.view.scene, color='black', font_size=12, anchor_x='left')

        # initial camera view center
        self.view.camera.center = (0, 0, 1)

        # Timer for updates
        self.timer = app.Timer(interval=1.0 / float(update_hz), connect=self._on_timer, start=False)

        # start timer after everything is ready
        self.timer.start()

        # hook close event to shutdown openvr gracefully
        self.canvas.events.close.connect(self._on_close)

    def _on_close(self, event):
        # stop timer and shutdown openvr
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            openvr.shutdown()
        except Exception:
            pass

    def _update_vr_data(self):
        # fetch poses
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        for device_idx in range(openvr.k_unMaxTrackedDeviceCount):
            if not poses[device_idx].bDeviceIsConnected or not poses[device_idx].bPoseIsValid:
                continue

            try:
                device_sn = self.vr_system.getStringTrackedDeviceProperty(
                    device_idx, openvr.Prop_SerialNumber_String
                )
            except Exception:
                continue

            if device_sn not in self.device_sn_to_pose_name_map:
                continue

            name = self.device_sn_to_pose_name_map[device_sn]
            pose_mat = np.zeros((4,4), dtype=float)
            # copy data: openvr returns a struct with .m (3x4)
            pose_mat[0:3, 0:4] = poses[device_idx].mDeviceToAbsoluteTracking.m
            pose_mat[3,3] = 1.0

            pos = pose_mat[0:3, 3].copy()
            R = pose_mat[0:3, 0:3].copy()

            # offsets for display
            pos[0] += self.x_offset
            pos[1] += self.y_offset

            self.positions[name] = pos
            self.orientations[name] = R

    def _set_marker(self, marker_obj, pos, color_rgba, size=12):
        marker_obj.set_data(np.array([pos]), face_color=color_rgba, size=size, edge_color='black')

    def _set_axis_line(self, line_obj, start, vec):
        # line expects Nx3 points; for segments we provide shape (2,3) and connect='segments'
        pts = np.vstack([start, start + vec])
        line_obj.set_data(pts)

    def _on_timer(self, event):
        # fetch latest VR poses
        try:
            self._update_vr_data()
        except Exception as e:
            # don't crash visualizer if OpenVR hiccups
            print("VR update error:", e, file=sys.stderr)
            return

        # update markers
        root_pos = self.positions["root"]
        target_pos = self.positions["target"]
        self._set_marker(self.marker_root, root_pos, (0.2, 0.4, 1.0, 1.0), size=14)
        self._set_marker(self.marker_target, target_pos, (1.0, 0.3, 0.3, 1.0), size=14)

        # update connection line
        pts = np.vstack([root_pos, target_pos])
        self.conn_line.set_data(pts)

        # update axis lines and tips
        for name in ["root", "target"]:
            pos = self.positions[name]
            R = self.orientations[name]
            for i, axis in enumerate(["x","y","z"]):
                vec = R[:, i] * self.quiver_len
                self._set_axis_line(self.axis_lines[name][axis], pos, vec)
                # tip marker
                tip_pos = pos + vec
                self.tips[name][i].set_data(np.array([tip_pos]), face_color=[(1,0,0,1),(0,1,0,1),(0,0,1,1)][i], size=8)

            # update labels near each tracker
            if name == "root":
                self.text_root.pos = root_pos + np.array([0.02, 0.02, 0.02])
            else:
                self.text_target.pos = target_pos + np.array([0.02, 0.02, 0.02])

        # optionally: adjust camera center to midpoint (uncomment if desired)
        # midpoint = (root_pos + target_pos) / 2.0
        # self.view.camera.center = midpoint

        # request redraw (vispy will schedule efficient GPU redraw)
        self.canvas.update()

    def run(self):
        print("Vispy visualizer running. Close the window to exit.")
        app.run()

if __name__ == "__main__":
    vis = ViveVisualizerVispy(x_offset=2.0, y_offset=4.0, quiver_len=0.25, update_hz=60.0)
    vis.ru
