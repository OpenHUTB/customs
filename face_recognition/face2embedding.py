from face_app import *

app = FaceRecognition(1)
app.face_to_embedding("face_dataset", save_face=False)