#-----------import------------------


from mimetypes import knownfiles
from msilib import knownbits
import numpy as np
import tkinter as T 
from  tkinter import messagebox as m
from PIL import Image
from PIL import ImageTk
import face_recognition
from sklearn import svm
import os
from tkinter import filedialog
import cv2

#---------------------------------------------------------------window design
w= T.Tk()
message = T.Label(
    w, text="Face-Recognition-System Attendenc", fg="black", width=50,
    height=3, font=('times', 30, 'bold'))
 
message.place(x=200, y=20)


#------logo for  page
im=ImageTk.PhotoImage(file='logo page.jpg')
w.iconphoto(False, im)
#----image logo in page

im0=ImageTk.PhotoImage(file='11.png')
r=T.Label(image=im0, width=600,
    height=350, font=('times', 1, 'bold'))
r.place(x=500,y=165)

 #-------trining
takeImg = T.Button(w, text="Training",
                    command="", fg="white", bg="black",
                    width=20, height=3, activebackground="green",
                    font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)


#-----SVM button
svm = T.Button(w, text="Svm",
                     command=lambda:svm(), fg="white", bg="black",
                     width=20, height=3, activebackground="green",
                     font=('times', 15, ' bold '))
svm.place(x=500, y=500)

############################################
def kn():
    video_capture = cv2.VideoCapture(0)
    harry_image = face_recognition.load_image_file("image/h.JPG")
    harry_face_encoding = face_recognition.face_encodings(harry_image)[0]
    
    #ron_image = face_recognition.load_image_file("image/j.jpg")
    #ron_face_encoding = face_recognition.face_encodings(ron_image)[0]

    hermione_image = face_recognition.load_image_file("image/m.JPG")
    hermione_face_encoding = face_recognition.face_encodings(hermione_image)[0]

    known_face_encodings = [
    harry_face_encoding,
    #ron_face_encoding,
    hermione_face_encoding
    ]
    known_face_names = [
    "harry potter",
    #"Ron Weasly",
    "Hermione Jane"
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:  
            # Grab a single frame of video
       ret, frame = video_capture.read()

    # Only process every other frame of video to save time
       if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
         s_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
         r_frame = s_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
         face_locations = face_recognition.face_locations(r_frame)
         face_encodings = face_recognition.face_encodings(r_frame, face_locations)

         face_names = []
         for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

       process_this_frame = not process_this_frame


    # Display the results
       for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
         top *= 4
         right *= 4
         bottom *= 4
         left *= 4

        # Draw a box around the face
         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
         font = cv2.FONT_HERSHEY_DUPLEX
         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
         cv2.imshow('video', frame)

    # Hit 'q' on the keyboard to quit!
       if cv2.waitKey(0) & 0xFF == ord('q'):
          break

# Release handle to the webcam
       video_capture.release()
       cv2.destroyAllWindows()  

#----knn button
knn= T.Button(w, text="knn",
                     command=kn , fg="white", bg="black",
                     width=20, height=3, activebackground="green",
                     font=('times', 15, ' bold '))
knn.place(x=800, y=500)



#------excel button
ex = T.Button(w, text="attendence",
                     command="", fg="white", bg="black",
                     width=20, height=3, activebackground="green",
                     font=('times', 15, ' bold '))
ex.place(x=1100, y=500)


#------close button
quitWindow = T.Button(w, text="close",
                       command=w.destroy, fg="black", bg="white",
                       width=10, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=700, y=650)
 #-----------------------------------------------end button


#-----svm class
def svm():
    file=filedialog.askopenfilename()
    f=open(file,'r')
    print(f.read())
    f.close()

#----------------



w.title("Student Attendence System")
w.geometry('1500x1500')
w.mainloop()