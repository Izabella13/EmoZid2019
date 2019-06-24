#Import required modules
import cv2
import dlib
#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
#video_capture.set(3, 800)
#video_capture.set(4, 600)
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image
    for k, d in enumerate(detections): #For each detected face
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        shape = predictor(clahe_image, d) #Get coordinatesfor i in range(49,68): #There are 68 landmark points on each face
        for i in range(0,17): #There are 68 landmark points on each face    
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (255,0,255), 1)
        for i in range(18,26): #There are 68 landmark points on each face    
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (255,255,255), 1)
        for i in range(27,36): #There are 68 landmark points on each face    
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0,255,0), 1)
        for i in range(36,48): #There are 68 landmark points on each face    
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (255,0,0), 1)
        for i in range(49,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0,0,255), 1)
        cv2.circle(frame, (shape.part(30).x, shape.part(30).y), 2, (0,255,255), 1)    
            #For each point, draw a red circle with thickness2 on the original frame
    cv2.imshow('camera', frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
video_capture.release()
cv2.destroyAllWindows()