# Face Detection Script

This Python script processes videos to extract faces (both realistic and anime) and saves them along with full-frame screenshots. It uses GPU acceleration to achieve high performance for face detection and is ideal for tasks involving character or face extraction from video files.

## Features

- **Realistic Face Detection:** Uses `face_recognition` (HOG or CNN-based) for detecting human faces in videos.
- **Anime Character Detection:** Leverages YOLOv8 models trained specifically for anime-style characters.
- **Full-Frame Screenshots:** Saves every video frame as a screenshot in addition to detected faces.
- **Optimized for GPUs:** Ensures faster processing with CUDA-enabled libraries.
- **Dependency Check:** Automatically checks for required Python packages and provides installation instructions if dependencies are missing.
- **Organized Output:** Processed data is saved in structured folders for easier access and management.

## Requirements

- Python 3.8+
- A CUDA-capable GPU for optimal performance.
- YOLOv8 model for anime detection (e.g., `AniRef40000-l-epoch50.pt`).

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model for anime detection from [SoulflareRC/AniRef-yolov8](https://github.com/SoulflareRC/AniRef-yolov8) and place it in the `model` folder within your working directory. For example:
   ```
   /ScreenCap/model/AniRef40000-l-epoch50.pt
   ```

## Folder Structure

Ensure the following folder structure exists within your working directory:

```
/ScreenCap
├── input          # Input videos folder
├── Work             # Temporary working directory
│   ├── screencaps   # Stores full-frame screenshots
│   ├── faces        # Stores cropped faces/characters
├── output          # Output directory for processed results
└── model            # YOLOv8 model files
```

## Usage

1. Place video files in the `input` folder.
2. Run the script:
   ```bash
   python face_detection.py
   ```
3. Select the detection mode:
   - `1` for realistic face detection (HOG/CNN)
   - `2` for anime character detection (YOLOv8)
4. Processed results will be saved in the `output` folder.

## Outputs

- **Screencaps Folder:** Contains full-frame screenshots for each frame in the video.
- **Faces Folder:** Contains cropped images of detected faces or characters.
- Each processed video gets its own subfolder within the `output` directory.

## Example

1. **Input Video:** `example_video.mp4` placed in the `input` folder.
2. **Output Structure:**

```
/ScreenCap/output/example_video
├── screencaps
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   └── ...
├── faces
│   ├── anime_face_0_0.jpg
│   ├── anime_face_1_0.jpg
│   └── ...
```

## Key Functions

### `detect_faces_realistic`
Detects human faces in frames using `face_recognition` and saves the cropped faces to the `faces` folder.

### `detect_faces_anime`
Detects anime characters using YOLOv8 and saves the cropped characters to the `faces` folder.

### `process_videos`
Processes all video files in the `input` folder and organizes results in the `output` folder.

## Dependency Check
The script automatically checks for the following dependencies:
- `opencv-python`
- `face_recognition`
- `Pillow`
- `tqdm`
- `ultralytics`

If any dependencies are missing, the script will output the necessary `pip install` commands.

## Notes

- Ensure that the YOLOv8 model is compatible with the script.
- GPU acceleration is highly recommended for faster processing.
- Customize the confidence threshold in the `detect_faces_anime` function if needed.

## License

This project is licensed under the GNU General Public License (GPL). See the `LICENSE` file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- `face_recognition` library for human face detection
- Anime-specific YOLOv8 models from [SoulflareRC/AniRef-yolov8](https://github.com/SoulflareRC/AniRef-yolov8)
- Community contributors for datasets and tools.
