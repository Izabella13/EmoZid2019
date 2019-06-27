import cv2
import dlib
import pickle
import numpy as np
from time import sleep
from PIL import ImageFont, ImageDraw, Image
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        xlist = []
        ylist = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        meannp = np.asarray((shape.part(30).x, shape.part(30).y))
        x30 = [(x - shape.part(30).x) for x in xlist]  # Calculate distance centre <-> other points in both axes
        y30 = [(y - shape.part(30).y) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(x30, y30, xlist, ylist):
            #landmarks_vectorised.append(w)
            #landmarks_vectorised.append(z)
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
        landmarks_vectorised[:]=landmarks_vectorised[:]/landmarks_vectorised[27]
            #landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
    if len(detections) == 0:
        #xlist=0
        #ylist=0
        #meannp=np.asarray((0, 0))
        landmarks_vectorised=0
        detections=0
        face_descriptor=0
        #xlist,ylist,meannp,
    return landmarks_vectorised,detections, face_descriptor
emo=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх']
font= ImageFont.truetype("DejaVuSans.ttf", 32)
f=open('svm_dat.dat','rb')
clf=pickle.load(f)
f.close
video_capture = cv2.VideoCapture(0) #Webcam object
video_capture.set(3, 380)
video_capture.set(4, 240)
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
emo_sm=0
alfa=0.9
while True:
    ret, frame = video_capture.read()
    ''' gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray) '''
    l_mv, det,f_n = get_landmarks(frame)
    #l_mv, det = get_landmarks(clahe_image)
    if l_mv!=0:
        dat=[]
        dat.append(f_n)
        #dat.append(l_mv) 
        em_n=clf.predict(dat)
        emo_sm=alfa*(em_n-1)+(1-alfa)*emo_sm
        print(em_n,emo[round(float(em_n-1))],emo_sm+1)
        cv2.rectangle(frame, (det[0].left(), det[0].top()), (det[0].right(), det[0].bottom()), (0, 0, 255), 2)
        #cv2.putText(frame, emo[round(float(emo_sm))], (det[0].left(), det[0].top()), font, 2, (0, 255, ), 2)

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text( (det[0].left(), det[0].top()-40),  emo[round(float(emo_sm))], font=font, fill=(0,255,0))
        frame = np.array(img_pil)

    cv2.imshow('camera', frame) #Display the frame
    #sleep(0.3)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
video_capture.release()
cv2.destroyAllWindows()