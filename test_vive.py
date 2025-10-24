#!/usr/bin/env python3

import sys
import numpy as np
import openvr
import time

class ViveTracker():
    def __init__(self):
        self.vr_system = openvr.init(openvr.VRApplication_Other)

        self.device_sn_to_pose_name_map = {
            "LHR-8C30BD01": "root",
            "LHR-1FB29FC6": "target",
        }

    def __del__(self):
        openvr.shutdown()

    def run(self):
        print_info = True
        while True:
            self.device_data_list = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )


            if print_info:
                print("Device info:")

            for device_idx in range(openvr.k_unMaxTrackedDeviceCount):
                self.process_single_device_data(device_idx, print_info)

            if print_info:
                print_info = False
                
            time.sleep(0.01)

    def process_single_device_data(self, device_idx, print_info=False):
        if not self.device_data_list[device_idx].bDeviceIsConnected:
            return

        device_type = self.vr_system.getTrackedDeviceClass(device_idx)
        device_sn = self.vr_system.getStringTrackedDeviceProperty(device_idx, openvr.Prop_SerialNumber_String)
        is_pose_valid = self.device_data_list[device_idx].bPoseIsValid
        pose_matrix_data = self.device_data_list[device_idx].mDeviceToAbsoluteTracking

        if print_info:
            type_map = {
                openvr.TrackedDeviceClass_GenericTracker: "Tracker",
                openvr.TrackedDeviceClass_Controller: "Controller",
                openvr.TrackedDeviceClass_TrackingReference: "BaseStation",
                openvr.TrackedDeviceClass_HMD: "HMD"
            }
            device_type_str = type_map.get(device_type, "Unknown")
            print(f"  - {device_type_str}: {device_sn}")

        if device_sn not in self.device_sn_to_pose_name_map or not is_pose_valid:
            return

        pose_name = self.device_sn_to_pose_name_map[device_sn]

        pose_mat = np.zeros((4, 4))
        pose_mat[0:3, 0:4] = pose_matrix_data.m
        pose_mat[3, 3] = 1.0

        print(pose_name)
        print(pose_mat[0:3,3])


if __name__ == "__main__":
    vive_tracker = ViveTracker()
    vive_tracker.run()