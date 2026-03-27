import time

import cv2
import matplotlib.pylab as plt
import numpy as np
from matplotlib import patches


class RolloutVisualPrompts:
    def set_args_for_visual_prompts(self, parser):
        parser.add_argument("--visual_prompt", action="store_true")
        parser.add_argument("--save_visual_prompt", type=str, default=None)
        parser.add_argument("--load_visual_prompt", type=str, default=None)
        parser.add_argument(
            "--visual_prompt_types",
            type=str,
            nargs="+",
            choices=["p", "b"],
            default=None,
        )
        parser.add_argument("--visual_prompt_point_size", type=float, default=100.0)
        parser.add_argument("--visual_prompt_line_width", type=float, default=5.0)

    def setup_variables_for_visual_prompts(self):
        if self.args.visual_prompt and self.args.load_visual_prompt is None:
            if self.args.visual_prompt_types is None:
                raise ValueError("visual_prompt_types required")

        if self.args.load_visual_prompt is None:
            self.visual_prompts_all = {}
        else:
            print(
                f"[{self.__class__.__name__}] Load the visual prompts: {self.args.load_visual_prompt}"
            )
            loaded = np.load(self.args.load_visual_prompt, allow_pickle=True)
            self.visual_prompts_all = loaded["visual_prompts_all"].item()

    def reset_variables_for_visual_prompts(self):
        self.visual_prompts = None

    def _get_prompt_colors(self, n):
        return plt.cm.Set1(np.linspace(0, 1, n + 1))[:n]

    def _render_frame(self, img, pts, types, colors, point_size, line_width):
        out = img.copy()

        for i in range(len(pts)):
            p = pts[i]
            t = types[i]
            c = (np.array(colors[i][:3]) * 255).astype(np.uint8).tolist()

            if t == "p":
                cv2.circle(
                    out,
                    (int(p[0]), int(p[1])),
                    int(point_size // 10),
                    c,
                    -1,
                )
            else:
                cv2.rectangle(
                    out,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    c,
                    int(line_width),
                )

        return out

    def _interactive_annotate(self, image, types):
        import tkinter as tk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        root = tk.Tk()
        root.withdraw()

        results = []

        for i, t in enumerate(types):
            done = [False]
            data = {"val": None}
            start = [None]
            rect = [None]

            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{i+1}/{len(types)}: {'point' if t=='p' else 'box'}")

            win = tk.Toplevel(root)
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            def press(e):
                if e.xdata is None:
                    return
                if t == "p":
                    data["val"] = (e.xdata, e.ydata, np.nan, np.nan)
                    done[0] = True
                    win.destroy()
                else:
                    start[0] = (e.xdata, e.ydata)

            def motion(e):
                if t != "b" or start[0] is None or e.xdata is None:
                    return
                if rect[0] is not None:
                    rect[0].remove()

                x1, y1 = start[0]
                x2, y2 = e.xdata, e.ydata

                r = patches.Rectangle(
                    (min(x1, x2), min(y1, y2)),
                    abs(x2 - x1),
                    abs(y2 - y1),
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(r)
                rect[0] = r
                canvas.draw()

            def release(e):
                if t != "b" or start[0] is None or e.xdata is None:
                    return
                x1, y1 = start[0]
                x2, y2 = e.xdata, e.ydata

                data["val"] = (
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                )
                done[0] = True
                win.destroy()

            fig.canvas.mpl_connect("button_press_event", press)
            fig.canvas.mpl_connect("motion_notify_event", motion)
            fig.canvas.mpl_connect("button_release_event", release)

            win.protocol(
                "WM_DELETE_WINDOW", lambda: (done.__setitem__(0, True), win.destroy())
            )

            while not done[0]:
                try:
                    root.update()
                    time.sleep(0.05)
                except tk.TclError:
                    break

            if data["val"] is None:
                root.destroy()
                return None

            results.append(data["val"])

        root.destroy()
        return results

    def set_visual_prompts(self):
        if not self.args.visual_prompt:
            return

        world_idx = self.data_manager.world_idx

        if self.args.load_visual_prompt is not None:
            self.visual_prompts = self.visual_prompts_all[world_idx]
            return

        vp = {}

        print("=== Visual prompt annotation mode ===")

        for cam in self.camera_names:
            img = self.info["rgb_images"][cam]
            img = cv2.resize(img, self.model_meta_info["data"]["image_size"])

            pts = self._interactive_annotate(img, self.args.visual_prompt_types)
            if pts is None:
                raise RuntimeError("annotation aborted")

            vp[cam] = {
                "points": np.array(pts, dtype=np.float32),
                "types": np.array(self.args.visual_prompt_types),
            }

        self.visual_prompts = vp

        if self.args.save_visual_prompt is not None:
            self.visual_prompts_all[world_idx] = vp

    def save_visual_prompts(self):
        if self.args.save_visual_prompt is None:
            return

        print(
            f"[{self.__class__.__name__}] Save the visual prompts: {self.args.save_visual_prompt}"
        )
        np.savez(
            self.args.save_visual_prompt,
            visual_prompts_all=np.array(self.visual_prompts_all, dtype=object),
        )

    def overlay_visual_prompts(self, cam, image):
        if not self.args.visual_prompt:
            return image

        if self.visual_prompts is None:
            self.set_visual_prompts()

        data = self.visual_prompts[cam]

        # --- scale correction ---
        H_orig, W_orig, _ = self.info["rgb_images"][cam].shape
        H_new, W_new, _ = image.shape
        scale = min(W_new / W_orig, H_new / H_orig)

        point_size = self.args.visual_prompt_point_size * scale
        line_width = self.args.visual_prompt_line_width * scale

        colors = self._get_prompt_colors(len(data["points"]))

        return self._render_frame(
            image,
            data["points"],
            data["types"],
            colors,
            point_size,
            line_width,
        )
