from fastapi import FastAPI, File, UploadFile
import cv2
import os
from pathlib import Path

app = FastAPI()

# Directories for saving files
IMAGE_DIR = "captured_images"
DOCUMENT_DIR = "uploaded_documents"
IMAGE_PATH = f"{IMAGE_DIR}/latest_photo.jpg"  # Single file for storing latest photo

# Ensure directories exist
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(DOCUMENT_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/capture")
async def capture_photo():
    """Opens camera, captures a photo, and overwrites the previous image."""
    cap = cv2.VideoCapture(0)  # Open default camera (0)
    if not cap.isOpened():
        return {"error": "Camera not accessible"}
    
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("Press 'Q' to capture", frame)
        
        # Wait for key press
        k = cv2.waitKey(1)
        if k == ord('q'):  # Press 'q' to capture and save the image
            cv2.imwrite(IMAGE_PATH, frame)  # Overwrite previous image
            cv2.destroyAllWindows()
            break
    
    cap.release()  # Release the camera

    if not ret:
        return {"error": "Failed to capture image"}

    return {"message": "Photo captured successfully", "path": IMAGE_PATH}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a document and saves it."""
    file_location = f"{DOCUMENT_DIR}/{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    return {"message": "File uploaded successfully", "path": file_location}
