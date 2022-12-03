import customtkinter
import sqlite3
import tkinter as tk
import os
import numpy as np
import cv2
from PIL import Image

#classifiers------------
face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_eye.xml')

#capture and font------
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

#recognizer ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

#functions---------

#get profile from sql
def getProfile(Id):
    conn=sqlite3.connect("FaceBase")
    query="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

# train funct
def trainData(path):
        imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow('training', faceNp)
            cv2.waitKey(10)
        return np.array(IDs), faces
    


#stream funct
def stream():
    recognizer.read("trainer/training_data.yml")
    if not os.path.isfile("trainer/training_data.yml"):   
        print("Please train the data first !!")
        exit(0)
    while True:
    
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                profile=getProfile(nbr_predicted)
                if profile != None:
                    cv2.putText(img, "Name: "+str(profile[1]), (x, y+h+30), font, 0.4, (0, 0, 255), 1);
                    cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
                    cv2.putText(img, "Gender: " + str(profile[3]), (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
                    cv2.putText(img, "Authority: " +str(profile[4]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
            else:
                cv2.putText(img, "Name: Unknown", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Age: Unknown", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Gender: Unknown", (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Authority: Unknown", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);

        cv2.imshow('img', img)
        if(cv2.waitKey(1) == ord('q')):
            break
    cv2.destroyAllWindows()
    
    

#connection to db---------
def insertOrUpdate(id, name):
    
    conn =sqlite3.connect("FaceBase.db")
    #check if id already exists-------
    query = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(query)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if isRecordExist==1:
        query="UPDATE People SET Name="+str(name)+" WHERE ID="+str(id)
    else:
        query="INSERT INTO People(ID, Name) VALUES("+str(id)+","+str(name)+")"
    conn.execute(query)
    conn.commit()
    conn.close()

def takeData():
    cap = cv2.VideoCapture(0)
    id = entryID.get()
    name = entryName.get()
    insertOrUpdate(id, name)
    sample_number = 0
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            sample_number += 1

            if not os.path.exists('dataSet'):
                os.makedirs('dataSet')

            cv2.imwrite('dataSet/User.'+str(id)+"."+str(sample_number)+".jpg",  gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x-50,y-50), (x+w+50, y+h+50), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex-50, ey-50), (ex+ew+50, ey+eh+50), (0, 0, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        if(sample_number>30):
            cap.release()
            cv2.destroyAllWindows()
            trainData(path)
            Ids, faces = trainData(path)
            recognizer.train(faces, Ids)

            if not os.path.exists('trainer'): 
                os.makedirs('trainer')

            recognizer.save('trainer/training_data.yml')
            cv2.destroyAllWindows()
            break;    
            



#tkinter-----------
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("600x400")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20 , padx=20 ,fill="both" , expand =True)

label = customtkinter.CTkLabel(master=frame, text ="Face Recognition System", text_font=("Roboto"))
label.pack(pady=12 , padx=10)

entryID =customtkinter.CTkEntry(master=frame, placeholder_text="ID")
entryID.pack(pady=5, padx=10)

entryName =customtkinter.CTkEntry(master=frame, placeholder_text="Name")
entryName.pack(pady=5, padx=10)

entryAuth =customtkinter.CTkEntry(master=frame, placeholder_text="Authority")
entryAuth.pack(pady=5, padx=10)


button = customtkinter.CTkButton(master=frame , text ="Add New", command=takeData)
button.pack(pady=5, padx=10)

button = customtkinter.CTkButton(master=frame , text ="Detect Face", command=stream)
button.pack(pady=5, padx=10)

root.mainloop()