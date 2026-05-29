# Misc

## Data utilities
The file format pointed to by `<rmb_file>` can be either RmbData-Compact (`.rmb`) or RmbData-SingleHDF5 (`.hdf5`).

### Visualize demonstration data
Visualize the demonstration data by plotting it.

```console
$ uv run python ./VisualizeData.py <rmb_file>
```

### Switch internal format of demonstration data
Switch RMB format file between RmbData-Compact (`.rmb`) and RmbData-SingleHDF5 (`.hdf5`). The format is automatically determined from the file extension.

```console
$ uv run python ./SwitchRmbDataFormat.py <rmb_file_in> <rmb_file_out>
```

### Compare demonstration data
Compare the contents of the two RMB format files to see if they match.
Note that when converting images to mp4 files, lossy compression is applied to color images and quantization is applied to depth images, so RmbData-Compact contains some errors.

```console
$ uv run python ./CompareRmbData.py <rmb_file1> <rmb_file2>
```

### Refine demonstration data
Update the task description attribute in RMB format files. It accepts a path to a file or directory and automatically searches for relevant files. If the task description attribute exists and `--overwrite` is not specified, the value is not changed.

```console
$ uv run python ./RefineRmbData.py <rmb_file> --task_desc "<new_description>" [--overwrite]
```

## Visualization utilities
### Visualize camera images
Display the web camera image for recording the experiments.
```console
$ uv run python ./DisplayCameraImage.py --camera_name Webcam --resize_width 800 --win_xy 1000 400
```

Display the cropped camera image. This is useful for image cropping policies such as SARNN.
```console
$ uv run python ./DisplayCameraImage.py --camera_name RealSense --crop_size 280 280
```

A camera can also be specified by `--camera_id` instead of `--camera_name`.

## Video utilities
### Tile rollout videos
The input is a video consisting of a sequence of multiple rollouts, and the output is a tiled video of each rollout.
```console
$ uv run python ./TileRolloutVideos.py <input_video_path> --output_file_name <output_video_path> --task_success_list 1 0 1 0 1 1 --column_num 3
```
The options `--output_file_name`, `--task_success_list`, and `--column_num` can be omitted.

By default, the video separation times are automatically determined by detecting restarts of windows in the video, but can be specified explicitly by adding the `--task_period_list` option as follows:
```console
--task_period_list 00:00.00-00:11.00 00:14.00-00:27.50 00:30.20-00:42.50 00:45.70-00:58.50 01:01.70-01:13.50 01:16.70-01:28.00
```
