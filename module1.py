video_capture = cv2.VideoCapture(0) #Webcam object
#video_capture.set(3, 800)
#video_capture.set(4, 600)
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

