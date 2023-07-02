import cv2
from typing import List

def highlightFace(net, frame, conf_threshold=0.7):
    '''
    To detect face and highlight it with rectangle box
    '''
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def load_model_network():
    '''
    To load model and network using opencv dnn module
    with prototxt and caffemodel file 
    '''
    faceProto : str = "asset/model/opencv_face_detector.pbtxt"
    faceModel : str = "asset/model/opencv_face_detector_uint8.pb"
    ageProto : str = "asset/model/age_deploy.prototxt"
    ageModel : str = "asset/model/age_net.caffemodel"
    genderProto : str = "asset/model/gender_deploy.prototxt"
    genderModel : str = "asset/model/gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel,faceProto)
    ageNet = cv2.dnn.readNet(ageModel,ageProto)
    genderNet = cv2.dnn.readNet(genderModel,genderProto)
    return faceNet, ageNet, genderNet

def display_frame(video, padding):
    '''
    To display frame with rectangle box and text
    '''
    MODEL_MEAN_VALUES : tuple = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male','Female']
    ageList : List[str] =['(0-2)', '(4-6)', '(8-12)', \
        '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    faceNet, ageNet, genderNet = load_model_network()
    
    while cv2.waitKey(1)<0 :
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        
        resultImg,faceBoxes = highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face : List[tuple] = frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender : str = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age : str = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
            

