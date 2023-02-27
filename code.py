import cv2 as c

eye=c.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
face= c.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=c.VideoCapture(0)
while True:

    _,frame=cap.read()
    gray= c.cvtColor(frame,c.COLOR_BGR2GRAY)
    faces= face.detectMultiScale(gray,1.1,4)

    for x,y,w,h in faces:        
        c.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # c.circle(frame,(x,y),40,(0,255,0),3)
        roi_gray = gray[y:y+h,x:x+w] 
        roi_color= frame[y:y+h,x:x+w]
        eyes=eye.detectMultiScale(roi_gray)      
        for a,b,ce,d in eyes:
            c.rectangle(roi_color,(a,b),(a+ce,b+d),(0,0,255),5)
    c.imshow("video",frame)

    if c.waitKey(1)==27:
        break

cap.release()
