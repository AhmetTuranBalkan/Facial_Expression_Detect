# Facial Expression Classifier For CNN model
import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import Face_Recognition as objFaceRec

import model.CNN_model as objModel

Screen_name = 'Real_time Face expression recognition'

parser = argparse.ArgumentParser(description='Emotion recognition from webcam')
parser.add_argument('-testImage', help=('.'
"The model works well."))

args = parser.parse_args()
FACE_SHAPE = (48, 48)

model = objModel.fnBuildModel('cnn_model_weights_83.h5')

emotions     = ['Angry',
                'Fear',
                'Happy',
                'Sad',
                'Surprise',
                'Neutral']

color = [(0,0,255),(0,0,255),(255,0,0),(0,127,255),(130,0,75),(0,255,0)]

def fnRefreshFrame(frame, faceCoordinates, index):
    if faceCoordinates is not None:
        if index is not None:
            cv2.putText (frame, emotions[index], (faceCoordinates[0] - 10, faceCoordinates[1] - 15), cv2.FONT_HERSHEY_TRIPLEX, 2, color[index],
                         3, cv2.LINE_AA)
            objFaceRec.fnDrawFace (frame, faceCoordinates, color[index])
        else:
            objFaceRec.fnDrawFace (frame, faceCoordinates)

    cv2.imshow(Screen_name, frame)

# detects and predicts emotions from real time captured data
def fnShowAndDetect(capture):
    while (True):
        flag, frame = capture.read()
        faceCoordinates = objFaceRec.fnGetFaceCoordinates(frame)
        fnRefreshFrame(frame, faceCoordinates, None)
        
        if faceCoordinates is not None:
            face_img = objFaceRec.fnPreprocessImage(frame, faceCoordinates, face_shape=FACE_SHAPE)
            cv2.imshow(Screen_name, frame)
            objFaceRec.fnDrawFace(face_img, faceCoordinates)

            input_img = np.expand_dims(face_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print (emotions[index], 'prob:', max(result))
            fnRefreshFrame (frame, faceCoordinates, index)

        if cv2.waitKey(10) & 0xFF == 27:
            break

def fnGetCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
        
    return capture


def main():
    print("Enter main() function")
    
    if args.testImage is not None:
        img = cv2.imread(args.testImage)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, FACE_SHAPE)
        print(class_label[result[0]])
        sys.exit(0)

    showCam = 1

    capture = fnGetCameraStreaming()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(Screen_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(Screen_name, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    fnShowAndDetect(capture)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
