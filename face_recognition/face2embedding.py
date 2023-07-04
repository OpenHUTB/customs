from face_app import *

app = FaceRecognition(0)
app.face_to_embedding(r"face_dataset/photos", False)
# app.add_face("face_dataset/谭孝文_02.jpg")