import cv2
def generate():
    face_cascade = cv2.CascadeClassifier('C:\Users\HP\Desktop\haarcascade_frontalface_default.xml')
     
    camera=cv2.VideoCapture(0)
    count=0
    id=raw_input("enter the user id")
    while(1):
        ret,frame=camera.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            cv2.imwrite("C:\Users\HP\Desktop\data\dataSet\user."+str(id)+"."+str(count)+".jpg",f)
            count+=1
        cv2.imshow("camera",frame)
        k=cv2.waitKey(30)
        if k==27:
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    generate()
