import cv2

cap=cv2.VideoCapture(0)
classifier =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    rat,frame =cap.read()
    if rat:
        faces=classifier.detectMultiScale(frame)
        for face in faces:
            x,y,w,h=face
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)

        cv2.imshow("My window",frame)

        key=cv2.waitKey(20)

        if key==ord("b"):
            break

cap.release()
cv2.destroyAllWindows()
