import { Image, StyleSheet, View, Text, Pressable, ActivityIndicator } from 'react-native';

import { HelloWave } from '@/components/HelloWave';
import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Stack } from 'expo-router';
import { useCameraDevice, useCameraPermission, Camera, useFrameProcessor, useSkiaFrameProcessor, runAtTargetFps, PhotoFile } from 'react-native-vision-camera';
import { useEffect } from 'react';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { PaintStyle, rotate, Skia } from '@shopify/react-native-skia';
import { crop } from 'vision-camera-cropper';
import { OpenCV } from 'react-native-fast-opencv';
import { useSharedValue, Worklets } from 'react-native-worklets-core';
import { runOnJS } from 'react-native-reanimated';
import check_mark from '../../assets/images/check-mark.png'
import right_rotate from '../../assets/images/rotate-left.png'
import left_rotate from '../../assets/images/rotate-right.png'


import { useState, useRef } from 'react';
import { FontAwesome5 } from '@expo/vector-icons';
import { PermissionsAndroid, Alert } from 'react-native';

function GuideFrame({ predictRotation }: any) {
  const icon =
    Math.abs(predictRotation) <= 15
      ? check_mark
      : predictRotation > 0
        ? left_rotate
        : right_rotate;
  return (
    <Image
      style={styles.tinyLogo}
      source={icon}
    />
  )
}

function blobToBase64(blob: any) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      resolve(reader.result.split(',')[1]); // Extract base64 portion
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob); // This will encode the blob as a base64 string
  });
}

export default function HomeScreen() {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const TARGET_FPS = 2
  const { resize } = useResizePlugin();
  const camera = useRef<Camera>(null)
  const [predictRotation, setPredictRotation] = useState(0)
  const [photo, setPhoto] = useState<PhotoFile>()
  const [rotatedImgBase64, setRotatedImgBase64] = useState('')
  const [loading, setLoading] = useState(false); // New loading state

  const processImg = Worklets.createRunOnJS((img_base64: any) => {
    // console.log(img_base64)
    let body = {
      "img_base64": img_base64
    }
    // Send the image to the backend
    fetch('https://144998276bd6.ngrok.app/process-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }).then((response) => response.json()).then((data) => {
      // console.log(data.rotation_angle)
      let rotate_angle = data.rotation_angle
      if (rotate_angle == null) {
        rotate_angle = 0
      }
      setPredictRotation(rotate_angle)
    })
  })

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    runAtTargetFps(TARGET_FPS, () => {
      'worklet'

      const height = frame.height;
      const width = frame.width;

      const resized = resize(frame, {
        scale: {
          width: width,
          height: height,
        },
        pixelFormat: 'bgr',
        dataType: 'uint8',
      });

      const src = OpenCV.frameBufferToMat(height, width, 3, resized);

      const img_base64 = OpenCV.toJSValue(src).base64;
      processImg(img_base64);
    })
  }, [processImg])


  useEffect(() => {
    if (!hasPermission) {
      requestPermission()
    }
  }, [hasPermission])


  const onTakePicturePressed = async () => {
    setLoading(true);
    try {
      console.log('onTakePicturePressed')
      const photo = await camera.current?.takePhoto()
      const result = await fetch(`file://${photo?.path}`)
      const data = await result.blob();
      const img_base64 = await blobToBase64(data)
      // console.log(img_base64)
      fetch('https://144998276bd6.ngrok.app/rotate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "img_base64": img_base64 }),
      }).then((response) => response.json()).then((data) => {
        // console.log(data)
        setRotatedImgBase64(data.rotated_image)
      })
      setPhoto(photo)
    }
    catch (error) {
      // console.log(error)
    }
  }

  if (!hasPermission) {
    return (
      <View>
        <Stack.Screen options={{ headerShown: false }} />
        <Text>Camera permission not granted</Text>
      </View>
    )
  }

  if (device == null) {
    return (
      <View>
        <Stack.Screen options={{ headerShown: false }} />
        <Text>Camera not available</Text>
      </View>
    )
  }

  return (
    <>
      {
        photo ? (
          <View style={styles.container}>
            <Image
              source={{ uri: rotatedImgBase64 || `file://${photo?.path}` }}
              style={styles.image}
              onLoadEnd={() => {
                if (rotatedImgBase64 !== '') {
                  setLoading(false);
                }
              }}
            />
            <FontAwesome5
              name="arrow-left"
              size={20}
              color="white"
              style={styles.backButton}
              onPress={() => {
                setPhoto(undefined)
                setRotatedImgBase64('')
              }}
            />
            {loading && ( // Show loading indicator while processing
              <View style={styles.loadingOverlay}>
                <ActivityIndicator size="large" color="#00ff00" />
                <Text style={styles.loadingText}>Processing...</Text>
              </View>
            )}
          </View>

        ) : (
          <View style={styles.cameraContainer}>
            <GuideFrame predictRotation={predictRotation} />
            <Camera
              style={styles.camera}
              device={device}
              isActive={true}
              frameProcessor={frameProcessor}
              ref={camera}
              photo={true}
            />

            <Pressable
              style={styles.captureButton}
              onPress={onTakePicturePressed} />
          </View>
        )
      }
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000', // Đặt màu nền ổn định
  },
  photoContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative', // Đảm bảo các thành phần con nằm chính xác trong khung
  },
  image: {
    flex: 1,
    width: '100%', // Đảm bảo ảnh chiếm toàn bộ chiều rộng
    resizeMode: 'contain', // Điều chỉnh nếu cần để giữ tỷ lệ ảnh
  },
  backButton: {
    position: 'absolute',
    top: 30,
    left: 20,
    zIndex: 1, // Đảm bảo nút luôn nằm trên ảnh
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: 'white',
    fontSize: 16,
  },
  cameraContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  captureButton: {
    position: 'absolute',
    bottom: 50,
    width: 75,
    height: 75,
    backgroundColor: 'white',
    borderRadius: 37.5,
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
  },
  tinyLogo: {
    width: 100,
    height: 100,
    marginTop: 50,
    alignSelf: 'center',
  },
});
