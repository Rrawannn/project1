import cv2
import numpy as n
from datetime import datetime
import os
import face_recognition
import csv

video=cv2.VideoCapture(0)
h=face_recognition.load_image_file("image\h.JPG")
h_e=face_recognition.face_encodings(h)[0]
j=face_recognition.load_image_file("image\j.jpg")
j_1=face_recognition.face_encodings(j)[0]
m=face_recognition.load_image_file("image\m.JPG")
m_e=face_recognition.face_encodings(m)[0]

known_face_encoding=[
h_e,
j_1,
m_e
]

known_face_names =[
"hanry",
"jane",
"melano"
    ]

student= known_face_names.copy()
face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
c=now.strftime("%Y-%M-%D")
f=open (c+'.csv','w+',newline='')
lnwriter=csv.writer(f)
while True:
    # Grab a single frame of video
    __, frame = video.read()
    small_frame =cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    r_small_frame= small_frame[:,:,::,-1]
    if s:
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names=[]

        for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
           matches = face_recognition.compare_faces(known_face_encoding , face_encoding)
           name=""
           face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
           best=n.argmin(face_distance)
           if matches[best]:
            name = known_face_names[best]
           face_names.append(name)
           if name in known_face_names:
             if name is student:
                student.remove(name)
                print(student)
                c=now.strftime("%Y-%M-%S")
                lnwriter.writerow([name,c])

    cv2.imshow("attendence System ",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

video.release()
cv2.destroyAllWindows
f.close
        