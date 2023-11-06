import face_recognition
import cv2
import numpy as np
import dlib
import os
import time


##* to save the result lab_images 
folder_path = "Result_Lab_images"
video_capture = cv2.VideoCapture(0)

# Load a sample picture of Zekaria.

zekaria_image = face_recognition.load_image_file("known_people/zekaria.jpg")
zekaria_face_encoding = face_recognition.face_encodings(zekaria_image)[0]


# Loading a second sample picture of kate.
kate_image = face_recognition.load_image_file("known_people/kate.jpg")
kate_face_encoding = face_recognition.face_encodings(kate_image)[0]


#? collection known face encodings
known_face_encodings = [
    zekaria_face_encoding,
   kate_face_encoding,
]

known_face_names = [
    "Zekaria",
    "Kate"
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

   

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        ## print(f"To see what is coming from best match {best_match_index}") ##? this prints 0 if the match is true
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom),(0,255,0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 48), (right, bottom),  (0,255,0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        cv2.putText(frame, name, (left + 7, bottom - 7), font, 2.0, (255, 0, 0), 2)
        
        #? saving the images to the folder result_lab_images 

        cv2.imwrite(os.path.join(folder_path, f"{name}_{best_match_index}.jpg"), frame)

    
    
        
    # Display the resulting image
    cv2.imshow('Video', frame)
    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
