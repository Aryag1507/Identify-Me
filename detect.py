import cv2

def faceBox(faceNet, frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    arr = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            arr.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
    return frame, arr


faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(24-32)', '(38-45)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0) 

while True:
    ret, frame = video.read()
    frameNet, box = faceBox(faceNet,frame)
    for b in box:
        face = frameNet[b[1]:b[3], b[0]:b[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)
        genderNet.setInput(blob)
        genderPrediction = genderNet.forward()
        gender = genderList[genderPrediction[0].argmax()]
        ageNet.setInput(blob)
        agePrediction = ageNet.forward()
        age = ageList[agePrediction[0].argmax()]

        label = '{},{}'.format(gender, age)

        cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,255), 1)

    cv2.imshow("Age-Gender",frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()