import cv2 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

first_frame=None
#triggers the camera
video=cv2.VideoCapture(0) #in this brackets we can capture frm  mp4 files too
a=0
while True:
    a=a+1
    check,frame=video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converting to gray image
    canny=cv2.Canny(gray,30,100)     #removes noise and increases accuracy
    
    if first_frame is None:
        first_frame = frame
        continue
    # delta is difference between the first and next frame and continues
    delta_frame=cv2.absdiff(first_frame,frame)
    # threshold frame shows the object moving in white color
    threshold_frame= cv2.threshold(delta_frame,45,255,cv2.THRESH_BINARY)[1]

    threshold_frame=cv2.dilate(threshold_frame,None,iterations=5)

    faces = face_cascade.detectMultiScale(frame,
                                          scaleFactor = 1.2,
                                          minNeighbors = 5)
    
    if len(faces)==0:
        print("No faces found")
    else:
        
        #print ("Number of faces Detected: " +str(faces.shape[0]))
        
        for i in range(0,faces.shape[0]):
            print(faces[i])
                
            x,y,w,h = faces[i]
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.resize(detected_face, (256, 256)) #resize to 48x48
            #cv2.imwrite("Face_"+str(i+1)+".jpg",detected_face) -- for individual images 
            cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)
            cv2.imwrite('Frame1.jpg',frame)
            cv2.imshow('Frame1',frame)   
           
            
            cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
            cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.imwrite('Frame2.jpg',frame)  
            cv2.imshow('Frame2',frame)
            
        
    key=cv2.waitKey(1)

    if key==ord('q'):
        break
            


#print(a) # helps to identify how many frames are captured in the video
video.release() # releases the camera
cv2.destroyAllWindows()
