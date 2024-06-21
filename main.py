import pickle
import cv2
import cvzone
import face_recognition
import numpy as np
import pyttsx3
import streamlit as st 

st.title("Movie Face Recognition")
options=["Yüzüklerin Efendisi","Harry Potter","Karayip Korsanları"]
select=st.selectbox("Bir film seçin",options,)
col1,col2=st.columns(2)
with col1:
    play_button=st.button("Play")
with col2:
    stop_button=st.button("Stop")



print(select)
window=st.image([])
#cap.set(cv2.CAP_PROP_FPS, 20) 

# character audio function
def audio(character_name):
    engine=pyttsx3.init()
    engine.say(f"{character_name}")
    engine.runAndWait()
    
    
print("Loading encoding file")
if select=="Yüzüklerin Efendisi":
    cap=cv2.VideoCapture("Black Gates.mp4")
    file=open("EncodeFile1.p","rb")
elif select=="Harry Potter":
    cap=cv2.VideoCapture("Harry Potter 2.mp4")
    file=open("EncodeFile2.p","rb")
elif select=="Karayip Korsanları":
    cap=cv2.VideoCapture("POTC.mp4")
    file=open("EncodeFile3.p","rb")
encodeListKnowmwithNames=pickle.load(file)
file.close()
encodeListKnowm,characterNames=encodeListKnowmwithNames
print(characterNames)
print("Loaded encoding file")

skip_frames = 5  # Her 5 karede bir işlem yap
frame_count = 0

if play_button:
    while True :
        ret,frame=cap.read()
        
        frame_count += 1
        if frame_count % skip_frames == 0:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            imgS=cv2.resize(frame,(0,0),None,fx=0.25,fy=0.25)
            imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

            faceCurFrame=face_recognition.face_locations(imgS)
            encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

            for encodeface,faceloc in zip(encodeCurFrame,faceCurFrame):
                facedis=face_recognition.face_distance(encodeListKnowm,encodeface)
                print("facedis",facedis)
                matchIndex=np.argmin(facedis)
                if facedis[matchIndex] < 0.70:
                    charactername=characterNames[matchIndex]
                else :
                    charactername="Unknown"

                y1,x2,y2,x1=faceloc
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
                bbox=x1,y1,x2-x1,y2-y1
                cvzone.cornerRect(frame,bbox)
                cvzone.putTextRect(frame,charactername,(x1, y1 - 15))
                audio(charactername)
            window.image(frame)
            if stop_button:
                break
