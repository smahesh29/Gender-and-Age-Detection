#A Gender and Age Detection program by Mahesh Sawant

import cv2
import re
import math
import argparse
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    base_conf = 0
    #print(detections.shape[2])
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        #print(confidence)
        if confidence > base_conf:
            max_i = i
            base_conf = confidence

    x1=int(detections[0,0,max_i,3]*frameWidth)
    y1=int(detections[0,0,max_i,4]*frameHeight)
    x2=int(detections[0,0,max_i,5]*frameWidth)
    y2=int(detections[0,0,max_i,6]*frameHeight)
    faceBoxes.append([x1,y1,x2,y2])
    cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


#parser=argparse.ArgumentParser()
#parser.add_argument('--image')

#args=parser.parse_args()

def age_detect(input_image):
    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0 – 2)', '(4 – 6)', '(8 – 12)', '(15 – 20)', '(25 – 32)', '(38 – 43)', '(48 – 53)', '(60 – 100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(input_image if input_image else 0)
    #print(video)
    padding=20
    gender_lst = []
    age_lst = []

    while cv2.waitKey(1)<0 :
        hasFrame,frame=video.read()
        #print(frame.shape)
        if not hasFrame:
            cv2.waitKey()
            break
    
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        #print(len(faceBoxes))
        if not faceBoxes:
        #print("No face detected")
            gender_lst.append('No face detected')
            age_lst.append('No face detected')

        for faceBox in faceBoxes:
            #print(faceBox)
            face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
            #print(face)
            #print("face is None: ", np.any(face))
            if np.any(face) == True:
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                #print(genderPreds)
                #print(gender)
                
                gender_lst.append(gender)
        #print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                #print(agePreds)
                age=ageList[agePreds[0].argmax()]
                #print(age)
                age_lst.append(age)
        #print(f'Age: {age[1:-1]} years')

        #cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)
    #print(gender_lst)
    #print(age_lst)
    if len(gender_lst) ==1 :
        return gender_lst[0], age_lst[0]
    elif len(gender_lst) > 1:
        pattern = r"[–() ]"
        age_1 = [int(re.split(pattern, x)[1]) for x in age_lst]
        smallest_index = age_1.index(min(age_1))
        return gender_lst[smallest_index], age_lst[smallest_index]
    else:
        return None, None