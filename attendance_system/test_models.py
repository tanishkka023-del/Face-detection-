import os
import face_recognition_models

print("face_recognition_models package path:", face_recognition_models.__file__)
models_path = os.path.join(os.path.dirname(face_recognition_models.__file__), "models")
print("Expected models path:", models_path)
print("Files in models directory:", os.listdir(models_path))

# Attempt to load a model file directly
model_file = os.path.join(models_path, "dlib_face_recognition_resnet_model_v1.dat")
if os.path.exists(model_file):
    print(f"Model file {model_file} exists")
else:
    print(f"Model file {model_file} does not exist")
    