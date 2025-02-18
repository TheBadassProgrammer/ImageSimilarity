import sys
import numpy as np
import pyodbc
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model_util import DeepModel
import os
from tensorflow.keras.applications import VGG16, MobileNet, DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add your module path
sys.path.append('C:/Users/HP/Desktop/image-similarity/image-similarity')

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your React app's URL
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

# Ensure the directory exists
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

# Function to connect to the database
def connect_to_database():
    try:
        connection = pyodbc.connect(
            f"Driver={{{DATABASE_CONFIG['driver']}}};"
            f"Server={DATABASE_CONFIG['server']};"
            f"Database={DATABASE_CONFIG['database']};"
            f"Trusted_Connection={DATABASE_CONFIG['trusted_connection']};"
        )
        return connection
    except Exception as e:
        raise Exception(f"Database connection error: {e}")

# Function to fetch file paths for a product
def fetch_file_paths(product_code):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Query to fetch file paths for the given product code
        query = f"SELECT filePath FROM {DATABASE_CONFIG['table_name']} WHERE productCode = ?"
        cursor.execute(query, product_code)

        # Fetch all file paths
        file_paths = [row.filePath for row in cursor.fetchall()]
        return file_paths

    except Exception as e:
        print(f"Database error: {e}")
        return None

# Function to save file path and product code to the database
def save_to_database(product_code, file_path):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Insert query
        query = f"INSERT INTO {DATABASE_CONFIG['table_name']} (productCode, filePath) VALUES (?, ?)"
        cursor.execute(query, (product_code, file_path))
        connection.commit()

    except Exception as e:
        print(f"Database error: {e}")

# Image similarity processing class
class ImageSimilarityBatch:
    # Initialize models
    vgg_model = VGG16(weights='imagenet', include_top=False)  # VGG16 model
    mobilenet_model = MobileNet(weights='imagenet', include_top=False)  # MobileNet model
    densenet_model = DenseNet121(weights='imagenet', include_top=False)  # DenseNet121 model

    @staticmethod
    def process_images_batch(target_image_path, file_paths, model):
        """
        Processes a target image and compares it to an array of other images, calculating similarity scores.
        """
        try:
            # Select model and preprocessing function
            if model == 0:
                selected_model = ImageSimilarityBatch.mobilenet_model
                preprocess_function = mobilenet_preprocess_input
            elif model == 1:
                selected_model = ImageSimilarityBatch.vgg_model
                preprocess_function = vgg_preprocess_input
            elif model == 2:
                selected_model = ImageSimilarityBatch.densenet_model
                preprocess_function = densenet_preprocess_input
            else:
                raise ValueError("Invalid model parameter. Use 0 for MobileNet, 1 for VGG16, or 2 for DenseNet121.")

            # Preprocess the target image and extract features
            target_feature = ImageSimilarityBatch.extract_features(target_image_path, selected_model, preprocess_function)

            highest_similarity = 0
            best_match = None

            # Iterate through the list of file paths and calculate similarity
            for file_path in file_paths:
                # Preprocess each image and extract features
                feature = ImageSimilarityBatch.extract_features(file_path, selected_model, preprocess_function)

                # Calculate similarity
                similarity = ImageSimilarityBatch.calculate_similarity(target_feature, feature)

                # Track the highest similarity score
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = file_path

                # If similarity is above the threshold, return "Matched"
                if similarity > 0.75:
                    return {
                        "status": "Matched",
                        "similarity": float(similarity),  # Convert to Python float
                        "file_path": file_path,
                    }

            # If no matches found, return "Did not match" with the highest similarity
            return {
                "status": "Did not match",
                "highest_similarity": float(highest_similarity),
                "best_match": best_match,
            }

        except Exception as e:
            print(f"Error processing images: {e}")
            return {"status": "Error", "details": str(e)}

    @staticmethod
    def extract_features(image_path, model, preprocess_function):
        """
        Extracts features from an image file using the specified model and preprocessing function.
        """
        try:
            # Load the image with the target size expected by the model
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)  # Convert image to numpy array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_function(img_array)  # Preprocess image for the model

            # Pass the preprocessed image through the model to extract features
            features = model.predict(img_array)
            return features.flatten()  # Flatten the features to a 1D vector

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    @staticmethod
    def calculate_similarity(feature1, feature2):
        """
        Calculates the cosine similarity between two feature vectors.
        """
        from numpy.linalg import norm
        from numpy import dot

        if feature1 is None or feature2 is None:
            raise ValueError("Features cannot be None")

        # Compute cosine similarity
        return dot(feature1, feature2) / (norm(feature1) * norm(feature2))

# FastAPI Endpoints

@app.post("/check_similarity/")
async def check_similarity(product_code: str = Form(...), target_image: UploadFile = File(...), model: int = Form(...)):
    """
    Endpoint to check similarity for the given product_code and target image.
    """
    try:
        print(f"Received product_code: {product_code}")
        print(f"Received target_image: {target_image.filename}")
        print(f"Model selected: {model}")

        # Save the uploaded target image temporarily
        target_image_path = f"{IMAGE_STORAGE_PATH}{target_image.filename}"
        with open(target_image_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)

        # Fetch file paths for the product code
        file_paths = fetch_file_paths(product_code)

        if file_paths:
            # Process the images for similarity
            result = ImageSimilarityBatch.process_images_batch(target_image_path, file_paths, model)
            return JSONResponse({"status": result})
        else:
            return JSONResponse({"status": "No file paths found for the given product code"})

    except Exception as e:
        return JSONResponse({"status": "Error", "details": str(e)})

@app.post("/add_image/")
async def add_image(product_code: str = Form(...), image_file: UploadFile = File(...)):
    """
    Endpoint to add a new image for a given product_code.
    """
    try:
        # Save the uploaded image
        saved_file_path = f"{IMAGE_STORAGE_PATH}{image_file.filename}"
        with open(saved_file_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Save the file path and product code to the database
        save_to_database(product_code, saved_file_path)

        return JSONResponse({"status": "Image added successfully"})

    except Exception as e:
        return JSONResponse({"status": "Error", "details": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
