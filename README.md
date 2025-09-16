# DivyaDrishti

**DivyaDrishti** is a real-time vision assistant Android app for the visually impaired, using YOLOv8 and CameraX for robust object detection and accessible feedback.

---

## Features

- **Real-time Object Detection** — Fast and accurate detection of people, vehicles, signs, and other important objects using YOLOv8.
- **Dual Modes** — Auto-detect mode for continuous scanning; Tap-to-detect mode for instant feedback.
- **Accessible Feedback** — Spoken descriptions with Text-to-Speech and haptic vibration alerts.
- **Mobile AI Optimization** — Runs efficiently on modern Android devices using TensorFlow Lite and GPU acceleration.
- **User-friendly Design** — Clean interface and simple controls for maximum accessibility.

---

## Getting Started

### Prerequisites

- Android device with camera, Android 8.0 (API 26) or above
- Android Studio (recommended)

### Setup

1. **Clone the repository**
git clone https://github.com/Kanishk442/DivyaDrishti.git

text
2. **Open in Android Studio**
3. **Add model and labels**  
Place `yolov8n_float16.tflite` and `coco-labels-2014_2017.txt` into `app/src/main/assets/`.
4. **Build and Run**  
Connect your device and click **Run**.

### Usage

- Choose detection mode (Auto or Tap).
- Point the camera at your surroundings.
- Listen to spoken feedback and feel vibrational alerts for key object detections.

---

## Screenshots

_Add screenshots or demo GIFs here to showcase your app’s features._

---

## Roadmap

- [ ] Customizable detection targets
- [ ] Enhanced accessibility options
- [ ] Export detection history
- [ ] Offline support

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact & Contributions

**Author:** [Kanishk442](https://github.com/Kanishk442)  
Questions, feedback, and contributions are welcome!  
Open an issue or submit a pull request to collaborate.
