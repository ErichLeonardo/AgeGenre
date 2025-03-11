import cv2
import tkinter as tk
from tkinter import filedialog

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

image_path = select_image()
frame = cv2.imread(image_path)
if frame is None:
    print("Could not read the image")
    exit()

# Redimensionar la imagen para mejor visualización
frame = cv2.resize(frame, (800, 600))

padding = 20
resultImg, faceBoxes = highlightFace(faceNet, frame)

if not faceBoxes:
    print("No face detected")
else:
    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(x2 + padding, frame.shape[1] - 1), min(y2 + padding, frame.shape[0] - 1)
        
        face = frame[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        age_range = age[1:-1]

        print(f'Gender: {gender}')
        print(f'Age: {age_range}')

        if int(age_range.split('-')[0]) < 18:
            faceBlurred = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y1:y2, x1:x2] = faceBlurred
        
        # Mostrar la edad y género en la imagen
        label = f'{gender}, {age_range} years'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

cv2.imshow("Detected Age and Gender", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
