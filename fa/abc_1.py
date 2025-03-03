from fastapi import FastAPI, File, UploadFile
import cv2
import os
from pathlib import Path
import re
import cv2
import pytesseract

import cv2
import pytesseract

def image2text(image_path):
    image = cv2.imread(image_path)

    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to preprocess the image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.medianBlur(thresh, 3)  # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(thresh)
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
    
    # Display the preprocessed image (optional)
    cv2.imshow("Preprocessed Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_text(text):
    aadhaar_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    aadhaar_numbers = re.findall(aadhaar_pattern, text)
    return aadhaar_numbers
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
@app.post('/validate_Configuration_1')
async def validate_configuration_1(file : UploadFile=File(...)):
    b_file= await file.read()
    with open("abc.jpg",'wb') as f:
        f.write(b_file)
    return image2text("abc.jpg")

