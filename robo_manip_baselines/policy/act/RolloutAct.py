import os
import sys

import cv2
import matplotlib.pylab as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.common import RolloutBase, denormalize_data, TeleopRolloutBase


class RolloutAct(TeleopRolloutBase):
    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument(
            "--no_temp_ensem",
            action="store_true",
            help="whether to disable temporal ensembling of the inferred policy",
        )
        parser.add_argument(
            "--no_attention",
            action="store_true",
            help="whether to disable attention visualization",
        )

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(
            f"  - chunk size: {self.model_meta_info['data']['chunk_size']}, temporal ensembling: {not self.args.no_temp_ensem}"
        )

        # Construct policy
        DETRVAE.set_state_dim(self.state_dim)
        DETRVAE.set_action_dim(self.action_dim)
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])

        # Register hook to visualize attention images only if requested
        if not self.args.no_attention:

            def forward_hook(_layer, _input, _output):
                # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
                # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
                _layer.correlation_mat = _output[1][0].detach().cpu().numpy()

            for layer in self.policy.model.transformer.encoder.layers:
                layer.self_attn.correlation_mat = None
                layer.self_attn.register_forward_hook(forward_hook)

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        if self.args.no_plot:
            return
            
        if self.args.no_attention:
            # Create an improved single-row plot for cameras and actions
            num_cameras = len(self.camera_names)
            num_cols = max(num_cameras + 1, 4)
            fig_ax = plt.subplots(
                1,
                num_cols,
                figsize=(num_cols * 4, 4),
                dpi=100,
                squeeze=False,
                constrained_layout=True,
            )
        else:
            # Create the original two-row plot for attention maps
            fig_ax = plt.subplots(
                2,
                max(
                    len(self.camera_names) + 1,
                    len(self.policy.model.transformer.encoder.layers),
                ),
                figsize=(13.5, 6.0),
                dpi=60,
                squeeze=False,
                constrained_layout=True,
            )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.policy_action_buf = []
        self.policy_action_buf_history = []
        self._plot_text_handles = []

    def infer_policy(self):
        # Infer
        if (not self.args.no_temp_ensem) or (len(self.policy_action_buf) == 0):
            state = self.get_state()
            images = self.get_images()
            action = self.policy(state, images)[0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )
            if not self.args.no_temp_ensem:
                self.policy_action_buf_history.append(self.policy_action_buf)
                if (
                    len(self.policy_action_buf_history)
                    > self.model_meta_info["data"]["chunk_size"]
                ):
                    self.policy_action_buf_history.pop(0)

        # Store action
        if self.args.no_temp_ensem:
            action = self.policy_action_buf.pop(0)
        else:
            # Apply temporal ensembling to action
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(self.policy_action_buf_history)))
            exp_weights = exp_weights / exp_weights.sum()
            action = np.zeros(self.action_dim)
            for action_idx, _policy_action_buf in enumerate(
                reversed(self.policy_action_buf_history)
            ):
                action += exp_weights[::-1][action_idx] * _policy_action_buf[action_idx]
        self.policy_action = denormalize_data(action, self.model_meta_info["action"])
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def plot_images(self, axes):
        """Plots all camera feeds used by the policy onto the provided axes."""
        for camera_idx, camera_name in enumerate(self.camera_names):
            if camera_idx >= len(axes):
                break
            
            image = self.info["rgb_images"][camera_name].copy()

            # Draw crop rectangles if they are defined
            if hasattr(self, "parsed_camera_crops") and camera_name in self.parsed_camera_crops:
                x, y, w, h = self.parsed_camera_crops[camera_name]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            axes[camera_idx].imshow(image)
            axes[camera_idx].set_title(camera_name, fontsize=20)
            axes[camera_idx].axis("off")

    def draw_plot(self):
        if self.args.no_plot or not hasattr(self, "canvas"):
            return

        # Always clear previous text handles
        for handle in self._plot_text_handles:
            handle.remove()
        self._plot_text_handles.clear()
        
        # Clear all axes
        all_axes = np.ravel(self.ax)
        for ax in all_axes:
            ax.cla()
            ax.axis("off")

        if self.args.no_attention:
            # --- No-Attention Plotting ---
            self.plot_images(all_axes[:len(self.camera_names)])
            self.plot_action(all_axes[len(self.camera_names)])
        else:
            # --- Attention Plotting ---
            # Plot images and actions on the first row
            self.plot_images(self.ax[0, :len(self.camera_names)])
            self.plot_action(self.ax[0, len(self.camera_names)])

            # Plot attention maps on the second row
            attention_shape = (15, 20 * len(self.camera_names))
            for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
                if layer_idx < len(self.ax[1]) and layer.self_attn.correlation_mat is not None:
                    self.ax[1, layer_idx].imshow(
                        layer.self_attn.correlation_mat[2:, 1].reshape(attention_shape)
                    )
                    self.ax[1, layer_idx].set_title(
                        f"attention image ({layer_idx})", fontsize=20
                    )

        # --- Overlays (Common to Both Modes) ---
        reward = getattr(self, "reward", 0.0)
        done = getattr(self, "done", False)
        intervention_status = "FORCED" if self.force_intervention else ("ACTIVE" if self.is_intervention else "INACTIVE")
        status_text = f"Reward: {reward:.2f}\nDone: {done}\nIntervention: {intervention_status}"

        text_handle = self.fig.text(0.98, 0.98, status_text,
                      ha='right', va='top',
                      fontsize=12, color='black', weight='bold',
                      transform=self.fig.transFigure,
                      bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="grey", lw=1))
        self._plot_text_handles.append(text_handle)
        
        # --- Finalize and display ---
        self.canvas.draw()
        frame = np.asarray(self.canvas.buffer_rgba())
        cv2.imshow(self.policy_name, cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))

