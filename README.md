# Optimal Photo Capture Android Application

This Android application helps users capture photos at the optimal angle by providing real-time, on-screen guidance. Built using React Native, the app detects camera misalignment and automatically corrects the orientation if needed. The app leverages the **React Native Vision Camera** library for high-speed image streaming and integrates with a **FastAPI** backend powered by **OpenCV** for image processing. 

## Features

- **Real-Time Angle Guidance**: Visual arrows prompt users to adjust the camera angle for optimal photo alignment.
- **Automatic Image Correction**: If the user captures an image with an incorrect angle, the app automatically adjusts the orientation.
- **Cross-Platform Compatibility**: Built with React Native, ensuring smooth performance on Android.
- **Fast and Efficient Processing**: Uses FastAPI and OpenCV to process images quickly and return angle adjustments in real-time.

## Technologies Used

- **Frontend**: React Native, React Native Vision Camera
- **Backend**: FastAPI, OpenCV (Python)
- **Image Processing**: Base64 encoding for image transmission; edge and line detection to calculate alignment angle

## Getting Started

### Prerequisites

- **Node.js** and **npm**: Required to run the React Native app.
- **Python** and **FastAPI**: Required for backend image processing.
- **OpenCV**: Used in the backend for detecting image angles and adjustments.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sonnguyen112/Camera_Guide.git
   cd Camera_Guide
   ```

2. **Install frontend dependencies**:
   ```
   npm install
   ```

3. **Install backend dependencies**:
   ```
   cd backend
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the backend**:
   ```
   cd backend
   fastapi dev main.py 
   ```

2. **Start the React Native app**:
   Navigate to root directory of the project and run the following command:

   ```
   npx expo run:android
   ```