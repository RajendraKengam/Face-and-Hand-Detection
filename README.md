# Face Detection Project

A real-time face detection and hand tracking application using Python, OpenCV, and MediaPipe. This project captures video from your webcam, detects faces, tracks hands, and counts the number of raised fingers.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7+
- A webcam connected to your computer.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RajendraKengam/your-repository.git
    cd face-detection-project
    ```

2.  **Create and activate a virtual environment (recommended):**
    - On Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the MediaPipe model:**
    The hand tracking feature requires a model file. Download `hand_landmarker.task` from this link and place it in the root directory of the project.

## Usage

Run the `main.py` script to start the face detection application:

```bash
python main.py
```

A window will appear showing your webcam feed with detected faces highlighted. Press the **'q'** key to quit the application.

## How It Works

This application uses a pre-trained **Haar Cascade** model from OpenCV's library to perform face detection.

1.  The script initializes video capture from the default webcam.
2.  It loads the `haarcascade_frontalface_default.xml` classifier.
3.  In a loop, it reads frames from the webcam, converts them to grayscale, and performs face detection.
4.  For each detected face, a blue rectangle is drawn on the original frame.
5.  The processed frame is displayed in a window.
