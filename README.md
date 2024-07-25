# Emotion detection
This project implements real-time facial emotion detection using the `deepface` library and OpenCV. It captures video from the webcam. The emotion labels are displayed on the frames in real-time.
This is probably the shortest code to implement realtime emotion monitoring.


## Dependencies

- [deepface](https://github.com/serengil/deepface): A deep learning facial analysis library that provides pre-trained models for facial emotion detection. It relies on TensorFlow for the underlying deep learning operations.
- [OpenCV](https://opencv.org/): An open-source computer vision library used for image and video processing.

## Usage
### Initial steps:
- Git clone this repository Run: `git clone https://github.com/sujitnashik/Emotion-detection.git`
- Run: `cd Facial-Emotion-Recognition-using-OpenCV-and-Deepface`
1. Install the required dependencies:
   - You can use `pip install -r requirements.txt`
   - Or you can install dependencies individually:
      - `pip install deepface`
      - `pip install tf_keras`
      - `pip install opencv-python`

2. Download the Haar cascade XML file for face detection:
   - Visit the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and download the `haarcascade_frontalface_default.xml` file.

3. Run the code:
   - Execute the Python script.
   - Emotion labels will be displayed on the frames around detected faces.




