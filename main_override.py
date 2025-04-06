import sys
import numpy as np
import pyodbc
import shutil
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import VGG16, MobileNet, DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DATABASE_CONFIG = {
    "driver": "SQL Server",
    "server": "LAPTOP-U39SH27E\\NITIN",
    "database": "Products",
    "table_name": "XRayImages",
    "trusted_connection": "yes"
}

# Directory to store images
IMAGE_STORAGE_PATH = "C:/Users/HP/Desktop/XRayImages/"
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

# Function to connect to the database
def connect_to_database():
    return pyodbc.connect(
        f"Driver={{{DATABASE_CONFIG['driver']}}};"
        f"Server={DATABASE_CONFIG['server']};"
        f"Database={DATABASE_CONFIG['database']};"
        f"Trusted_Connection={DATABASE_CONFIG['trusted_connection']};"
    )

# Function to fetch file paths for a product
def fetch_file_paths(product_code):
    connection = connect_to_database()
    cursor = connection.cursor()
    query = f"SELECT filePath FROM {DATABASE_CONFIG['table_name']} WHERE productCode = ?"
    cursor.execute(query, product_code)
    return [row.filePath for row in cursor.fetchall()]

# Function to save file path and product code to the database
def save_to_database(product_code, file_path):
    connection = connect_to_database()
    cursor = connection.cursor()
    query = f"INSERT INTO {DATABASE_CONFIG['table_name']} (productCode, filePath) VALUES (?, ?)"
    cursor.execute(query, (product_code, file_path))
    connection.commit()

def align_and_extract_circuit(image_path):
    """
    Aligns the image using a perspective transform and extracts the blue circuit.
    """
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ensure at least one contour is found
    if not contours:
        return None

    # Sort by area and select the largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    phone_contour = contours[0]

    # Get the min area rectangle
    rect = cv2.minAreaRect(phone_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Function to order points correctly for perspective transform
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    ordered_box = order_points(box)

    # Compute width and height
    (tl, tr, br, bl) = ordered_box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    max_width = int(max(widthA, widthB))
    max_height = int(max(heightA, heightB))

    # Destination points for top-down view
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Get transformation matrix
    M = cv2.getPerspectiveTransform(ordered_box, dst_pts)

    # Apply perspective transform
    aligned = cv2.warpPerspective(orig, M, (max_width, max_height))

    # Convert to HSV and extract the blue circuit
    hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest blue contour (assuming it's the circuit)
    if contours:
        circuit_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(circuit_contour)  # Get rectangular bounding box

        # Crop the rectangular circuit region
        circuit = aligned[y:y+h, x:x+w]

        # Save and return the path of the extracted circuit
        cropped_path = image_path.replace(".jpg", "_cropped.jpg")
        cv2.imwrite(cropped_path, circuit)
        return cropped_path

    return None  # No blue circuit found


# Image similarity processing class
class ImageSimilarityBatch:
    vgg_model = VGG16(weights='imagenet', include_top=False)
    mobilenet_model = MobileNet(weights='imagenet', include_top=False)
    densenet_model = DenseNet121(weights='imagenet', include_top=False)

    @staticmethod
    def process_images_batch(target_image_path, file_paths, model):
        selected_model = [ImageSimilarityBatch.mobilenet_model, ImageSimilarityBatch.vgg_model, ImageSimilarityBatch.densenet_model][model]
        preprocess_function = [mobilenet_preprocess_input, vgg_preprocess_input, densenet_preprocess_input][model]
        target_feature = ImageSimilarityBatch.extract_features(target_image_path, selected_model, preprocess_function)
        highest_similarity, best_match = 0, None
        for file_path in file_paths:
            feature = ImageSimilarityBatch.extract_features(file_path, selected_model, preprocess_function)
            similarity = np.dot(target_feature, feature) / (np.linalg.norm(target_feature) * np.linalg.norm(feature))
            if similarity > highest_similarity:
                highest_similarity, best_match = similarity, file_path
            if similarity > 0.75:
                return {"status": "Matched", "similarity": float(similarity), "file_path": file_path}
        return {"status": "Did not match", "highest_similarity": float(highest_similarity), "best_match": best_match}

    @staticmethod
    def extract_features(image_path, model, preprocess_function):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_function(img_array)
        features = model.predict(img_array)
        return features.flatten()

# FastAPI Endpoints
@app.post("/check_similarity/")
async def check_similarity(product_code: str = Form(...), target_image: UploadFile = File(...), model: int = Form(...), isCrop: int = Form(0)):
    target_image_path = f"{IMAGE_STORAGE_PATH}{target_image.filename}"
    with open(target_image_path, "wb") as f:
        shutil.copyfileobj(target_image.file, f)
    if isCrop:
        cropped_path = align_and_extract_circuit(target_image_path)
        if cropped_path:
            target_image_path = cropped_path
    file_paths = fetch_file_paths(product_code)
    return JSONResponse({"status": ImageSimilarityBatch.process_images_batch(target_image_path, file_paths, model)})

@app.post("/add_image/")
async def add_image(product_code: str = Form(...), image_file: UploadFile = File(...), isCrop: int = Form(0)):
    saved_file_path = f"{IMAGE_STORAGE_PATH}{image_file.filename}"
    with open(saved_file_path, "wb") as f:
        shutil.copyfileobj(image_file.file, f)
    if isCrop:
        cropped_path = align_and_extract_circuit(saved_file_path)
        if cropped_path:
            saved_file_path = cropped_path
    save_to_database(product_code, saved_file_path)
    return JSONResponse({"status": "Image added successfully"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
