import argparse
import json

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    find_rmb_files,
)


def parse_argument():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="path to save the inferred safety limits",
    )
    # --- Relative Leeway Arguments ---
    parser.add_argument(
        "--position_leeway",
        type=float,
        default=0.0,
        help="Leeway percentage for joint position limits.",
    )
    parser.add_argument(
        "--velocity_leeway",
        type=float,
        default=0.0,
        help="Leeway percentage for joint velocity limits.",
    )
    parser.add_argument(
        "--eef_position_leeway",
        type=float,
        default=0.0,
        help="Leeway percentage for end-effector position boundaries (used for AABB fallback).",
    )
    parser.add_argument(
        "--eef_orientation_leeway",
        type=float,
        default=0.0,
        help="Leeway percentage for end-effector maximum orientation deviation.",
    )
    # --- Absolute Leeway Arguments ---
    parser.add_argument(
        "--position_leeway_abs",
        type=float,
        default=None,
        help="Absolute leeway for joint position limits (in degrees). Overrides percentage.",
    )
    parser.add_argument(
        "--velocity_leeway_abs",
        type=float,
        default=None,
        help="Absolute leeway for joint velocity limits (in degrees/sec). Overrides percentage.",
    )
    parser.add_argument(
        "--eef_position_leeway_abs",
        type=float,
        default=0.0,
        help="Uniform absolute leeway for end-effector position boundaries (in meters). Overrides percentage.",
    )
    parser.add_argument(
        "--eef_orientation_leeway_abs",
        type=float,
        default=0,
        help="Absolute leeway for end-effector orientation deviation (in degrees). Overrides percentage.",
    )
    # Per-axis absolute leeways
    parser.add_argument(
        "--eef_position_leeway_x_abs",
        type=float,
        default=0.0,
        help="Absolute leeway for the EEF X-axis (in meters). Overrides uniform absolute and percentage leeway.",
    )
    parser.add_argument(
        "--eef_position_leeway_y_abs",
        type=float,
        default=0.0,
        help="Absolute leeway for the EEF Y-axis (in meters). Overrides uniform absolute and percentage leeway.",
    )
    parser.add_argument(
        "--eef_position_leeway_z_abs",
        type=float,
        default=0.0,
        help="Absolute leeway for the EEF Z-axis (in meters). Overrides uniform absolute and percentage leeway.",
    )
    # --- Other Arguments ---
    parser.add_argument(
        "--preset",
        type=str,
        default='ur10e',
        choices=['ur5', 'ur5e', 'ur10', 'ur10e'],
        help="Apply default settings for a specific robot (e.g., 'ur5'). This will set conversion indices for the gripper.",
    )
    parser.add_argument(
        "--pos_deg_to_rad_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of joint indices where the POSITION is in degrees and should be converted to radians. Overrides preset.",
    )
    parser.add_argument(
        "--vel_deg_to_rad_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of joint indices where the VELOCITY is in degrees/sec and should be converted to rad/sec. Overrides preset.",
    )

    return parser.parse_args()


class InferSafetyLimits:
    """Infers safety limits from robot demonstration data."""
    def __init__(self, path, output_path, position_leeway, velocity_leeway,
                 eef_position_leeway, eef_orientation_leeway,
                 position_leeway_abs, velocity_leeway_abs,
                 eef_position_leeway_abs, eef_orientation_leeway_abs,
                 eef_position_leeway_x_abs, eef_position_leeway_y_abs, eef_position_leeway_z_abs,
                 pos_deg_to_rad_indices, vel_deg_to_rad_indices):
        self.path = path
        self.output_path = output_path
        # Relative leeways
        self.position_leeway = position_leeway
        self.velocity_leeway = velocity_leeway
        self.eef_position_leeway = eef_position_leeway
        self.eef_orientation_leeway = eef_orientation_leeway
        # Absolute leeways
        self.position_leeway_abs = position_leeway_abs
        self.velocity_leeway_abs = velocity_leeway_abs
        self.eef_position_leeway_abs = eef_position_leeway_abs
        self.eef_orientation_leeway_abs = eef_orientation_leeway_abs
        # Per-axis absolute leeways
        self.eef_position_leeway_x_abs = eef_position_leeway_x_abs
        self.eef_position_leeway_y_abs = eef_position_leeway_y_abs
        self.eef_position_leeway_z_abs = eef_position_leeway_z_abs
        # Other params
        self.pos_deg_to_rad_indices = pos_deg_to_rad_indices
        self.vel_deg_to_rad_indices = vel_deg_to_rad_indices
        self.safety_limits = {}

    def _load_data(self):
        """Loads and aggregates data from rmb files."""
        rmb_path_list = find_rmb_files(self.path)
        print(f"[{self.__class__.__name__}] Found {len(rmb_path_list)} files to process.")

        all_joint_pos, all_joint_vel = [], []
        all_eef_pos, all_eef_quat = [], []

        for rmb_path in tqdm(rmb_path_list, desc="Loading data"):
            try:
                with RmbData(rmb_path) as rmb_data:
                    if DataKey.MEASURED_JOINT_POS in rmb_data:
                        all_joint_pos.append(rmb_data[DataKey.MEASURED_JOINT_POS][:])
                    if DataKey.MEASURED_JOINT_VEL in rmb_data:
                        all_joint_vel.append(rmb_data[DataKey.MEASURED_JOINT_VEL][:])
                    if DataKey.MEASURED_EEF_POSE in rmb_data:
                        eef_poses = rmb_data[DataKey.MEASURED_EEF_POSE][:]
                        all_eef_pos.append(eef_poses[:, :3])
                        # Assuming w,x,y,z quaternion format for the raw data
                        all_eef_quat.append(eef_poses[:, 3:]) 
            except Exception as e:
                print(f"Warning: Could not process {rmb_path}: {e}")
                continue
        
        # Concatenate lists into numpy arrays if they contain data
        joint_pos_data = np.concatenate(all_joint_pos, axis=0) if all_joint_pos else np.array([])
        joint_vel_data = np.concatenate(all_joint_vel, axis=0) if all_joint_vel else np.array([])
        eef_pos_data = np.concatenate(all_eef_pos, axis=0) if all_eef_pos else np.array([])
        eef_quat_data = np.concatenate(all_eef_quat, axis=0) if all_eef_quat else np.array([])

        return joint_pos_data, joint_vel_data, eef_pos_data, eef_quat_data

    def _process_joint_limits(self, all_joint_pos, all_joint_vel):
        """Calculates joint position and velocity limits."""
        if all_joint_pos.size == 0:
            print("No joint position data found. Skipping joint limit inference.")
            return

        print("Inferring joint limits...")
        if self.pos_deg_to_rad_indices is not None:
            print(f"-> Converting joint POSITION indices {self.pos_deg_to_rad_indices} from degrees to radians.")
            all_joint_pos[:, self.pos_deg_to_rad_indices] = np.deg2rad(all_joint_pos[:, self.pos_deg_to_rad_indices])
        
        min_pos = np.min(all_joint_pos, axis=0)
        max_pos = np.max(all_joint_pos, axis=0)

        # Apply absolute or relative leeway for position
        if self.position_leeway_abs is not None and self.position_leeway_abs > 0:
            leeway = np.deg2rad(self.position_leeway_abs)
            print(f"-> Applying absolute position leeway of {self.position_leeway_abs} degrees ({leeway:.4f} rad).")
            min_pos_with_leeway = min_pos - leeway
            max_pos_with_leeway = max_pos + leeway
        else:
            pos_range = max_pos - min_pos
            min_pos_with_leeway = min_pos - pos_range * self.position_leeway
            max_pos_with_leeway = max_pos + pos_range * self.position_leeway
        
        self.safety_limits.update({
            "joint_position_min": min_pos_with_leeway.tolist(),
            "joint_position_max": max_pos_with_leeway.tolist(),
        })

        if all_joint_vel.size > 0:
            if self.vel_deg_to_rad_indices is not None:
                print(f"-> Converting joint VELOCITY indices {self.vel_deg_to_rad_indices} from deg/s to rad/s.")
                all_joint_vel[:, self.vel_deg_to_rad_indices] = np.deg2rad(all_joint_vel[:, self.vel_deg_to_rad_indices])

            max_vel = np.max(np.abs(all_joint_vel), axis=0)

            # Apply absolute or relative leeway for velocity
            if self.velocity_leeway_abs is not None and self.velocity_leeway_abs > 0:
                leeway = np.deg2rad(self.velocity_leeway_abs)
                print(f"-> Applying absolute velocity leeway of {self.velocity_leeway_abs} deg/s ({leeway:.4f} rad/s).")
                max_vel_with_leeway = max_vel + leeway
            else:
                max_vel_with_leeway = max_vel * (1 + self.velocity_leeway)
                
            self.safety_limits["joint_velocity_max"] = max_vel_with_leeway.tolist()

    def _process_eef_limits(self, all_eef_pos, all_eef_quat):
        """Calculates end-effector position and orientation limits."""
        if all_eef_pos.size == 0:
            print("No EEF data found. Skipping EEF limit inference.")
            return

        print("Inferring EEF limits...")
        # --- 1. EEF Positional Bounds (Convex Hull and AABB) ---
        try:
            # More accurate: Convex Hull defines the tightest convex workspace.
            hull = ConvexHull(all_eef_pos)
            self.safety_limits["eef_position_bounds_hull"] = hull.equations.tolist()
            print("-> Using Convex Hull for EEF position bounds.")
        except Exception as e:
            # Fallback: Axis-Aligned Bounding Box (AABB) if hull fails.
            print(f"-> Convex Hull failed ({e}). Falling back to AABB for EEF position bounds.")

        # Always compute AABB for copy-paste args and as a fallback
        min_eef_pos = np.min(all_eef_pos, axis=0)
        max_eef_pos = np.max(all_eef_pos, axis=0)
        
        # Apply absolute or relative leeway for EEF position with precedence
        leeway_vec = np.array([
            self.eef_position_leeway_x_abs,
            self.eef_position_leeway_y_abs,
            self.eef_position_leeway_z_abs
        ])

        if np.any(leeway_vec > 0):
            print(f"-> Applying per-axis absolute EEF position leeway of x={leeway_vec[0]:.4f}m, y={leeway_vec[1]:.4f}m, z={leeway_vec[2]:.4f}m.")
            min_eef_with_leeway = min_eef_pos - leeway_vec
            max_eef_with_leeway = max_eef_pos + leeway_vec
        elif self.eef_position_leeway_abs is not None and self.eef_position_leeway_abs > 0:
            print(f"-> Applying uniform absolute EEF position leeway of {self.eef_position_leeway_abs} meters.")
            min_eef_with_leeway = min_eef_pos - self.eef_position_leeway_abs
            max_eef_with_leeway = max_eef_pos + self.eef_position_leeway_abs
        else:
            if self.eef_position_leeway > 0:
                print(f"-> Applying relative EEF position leeway of {self.eef_position_leeway*100:.2f}%.")
            eef_pos_range = max_eef_pos - min_eef_pos
            min_eef_with_leeway = min_eef_pos - eef_pos_range * self.eef_position_leeway
            max_eef_with_leeway = max_eef_pos + eef_pos_range * self.eef_position_leeway

        self.safety_limits["eef_bound_min"] = min_eef_with_leeway.tolist()
        self.safety_limits["eef_bound_max"] = max_eef_with_leeway.tolist()

        # --- 2. EEF Orientation Target (Average Quaternion) ---
        # Computes the average quaternion (Karcher mean) via SVD/Eigen on the covariance matrix.
        M = np.dot(all_eef_quat.T, all_eef_quat)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
        if avg_quat[0] < 0: # Ensure canonical representation (w > 0)
            avg_quat *= -1
        self.safety_limits["eef_orientation_target"] = avg_quat.tolist() # Stored as [w, x, y, z]

        # --- 3. EEF Orientation Deviation ---
        # Fixed: Using 'scalar_first=True' for [w, x, y, z] input convention.
        R_all = Rotation.from_quat(all_eef_quat, scalar_first=True) 
        # R_avg represents the computed mean orientation
        R_avg = Rotation.from_quat(avg_quat, scalar_first=True) 
        
        # R_dev is the relative rotation from the mean to all samples
        # R_dev = R_all * R_avg.inv() calculates the rotation required to go from R_avg to R_all
        R_dev = R_all * R_avg.inv() 

        # Advanced: Covariance matrix describes the shape of rotational deviations
        dev_vectors = R_dev.as_rotvec()
        orientation_covariance = np.cov(dev_vectors, rowvar=False)
        self.safety_limits["eef_orientation_dev_covariance"] = orientation_covariance.tolist()
        
        # Simple: Max deviation angle for copy-paste args
        # The deviation angle is the norm of the rotation vector (in radians)
        dev_angles_rad = np.linalg.norm(dev_vectors, axis=1)
        max_dev_rad = np.max(dev_angles_rad)

        # Apply absolute or relative leeway for EEF orientation
        if self.eef_orientation_leeway_abs is not None and self.eef_orientation_leeway_abs > 0:
            leeway_rad = np.deg2rad(self.eef_orientation_leeway_abs)
            print(f"-> Applying absolute EEF orientation leeway of {self.eef_orientation_leeway_abs} degrees ({leeway_rad:.4f} rad).")
            max_dev_with_leeway = max_dev_rad + leeway_rad
        else:
            max_dev_with_leeway = max_dev_rad * (1 + self.eef_orientation_leeway)
        
        # FIX: Convert the final radian value to degrees before storing and printing.
        # This makes the output value consistent with the argument name and the consumer's expectation
        # (TeleopRolloutBase uses np.deg2rad on this argument).
        max_dev_with_leeway_deg = np.rad2deg(max_dev_with_leeway)
        self.safety_limits["eef_orientation_max_dev"] = max_dev_with_leeway_deg
    
    def _save_and_print_results(self):
        """Saves limits to a file and prints copy-pastable arguments."""
        if not self.safety_limits:
            print("No limits were inferred.")
            return
            
        if self.output_path:
            with open(self.output_path, "w") as f:
                json.dump(self.safety_limits, f, indent=4)
            print(f"\n✅ Safety limits saved to {self.output_path}")

        copy_paste_args = []
        if "joint_position_min" in self.safety_limits:
            min_pos_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["joint_position_min"])
            max_pos_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["joint_position_max"])
            copy_paste_args.append(f"--joint_position_min {min_pos_str}")
            copy_paste_args.append(f"--joint_position_max {max_pos_str}")
        if "joint_velocity_max" in self.safety_limits:
            max_vel_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["joint_velocity_max"])
            copy_paste_args.append(f"--joint_velocity_max {max_vel_str}")
        if "eef_bound_min" in self.safety_limits:
            min_eef_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["eef_bound_min"])
            max_eef_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["eef_bound_max"])
            target_orient_str = ' '.join(f'{v:.6f}' for v in self.safety_limits["eef_orientation_target"])
            # The value is now correctly in degrees for the command-line argument
            max_dev_str = f'{self.safety_limits["eef_orientation_max_dev"]:.6f}'
            copy_paste_args.extend([
                f"--eef-bound-min {min_eef_str}",
                f"--eef-bound-max {max_eef_str}",
                f"--eef-orientation-target {target_orient_str}",
                f"--eef-orientation-max-dev {max_dev_str}"
            ])
        
        if copy_paste_args:
            print("\n📋 Copy-pastable arguments:")
            print(" \\\n".join(copy_paste_args))

    def run(self):
        """Main execution function."""
        print(f"[{self.__class__.__name__}] Inferring safety limits from {self.path}")
        joint_pos, joint_vel, eef_pos, eef_quat = self._load_data()
        
        self._process_joint_limits(joint_pos, joint_vel)
        self._process_eef_limits(eef_pos, eef_quat)
        
        self._save_and_print_results()


if __name__ == "__main__":
    args = parse_argument()

    pos_indices = args.pos_deg_to_rad_indices
    vel_indices = args.vel_deg_to_rad_indices

    # Apply presets if custom indices aren't provided
    if args.preset and args.pos_deg_to_rad_indices is None and args.vel_deg_to_rad_indices is None:
        preset_name = args.preset.lower()
        if preset_name in ['ur5', 'ur5e', 'ur10', 'ur10e']:
            print(f"Applying preset for {args.preset}: Gripper at index 6 will be converted from degrees.")
            pos_indices = [6]
            vel_indices = [6]
    
    infer_limits = InferSafetyLimits(
        path=args.path, 
        output_path=args.output_path, 
        position_leeway=args.position_leeway, 
        velocity_leeway=args.velocity_leeway,
        eef_position_leeway=args.eef_position_leeway,
        eef_orientation_leeway=args.eef_orientation_leeway,
        position_leeway_abs=args.position_leeway_abs,
        velocity_leeway_abs=args.velocity_leeway_abs,
        eef_position_leeway_abs=args.eef_position_leeway_abs,
        eef_orientation_leeway_abs=args.eef_orientation_leeway_abs,
        eef_position_leeway_x_abs=args.eef_position_leeway_x_abs,
        eef_position_leeway_y_abs=args.eef_position_leeway_y_abs,
        eef_position_leeway_z_abs=args.eef_position_leeway_z_abs,
        pos_deg_to_rad_indices=pos_indices,
        vel_deg_to_rad_indices=vel_indices
    )
    infer_limits.run()
