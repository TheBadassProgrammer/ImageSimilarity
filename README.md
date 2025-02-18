# Image Similarity Project

This project is an **image similarity application** built using **FastAPI** for comparing images based on their features. It leverages **MobileNet** as a deep learning model to extract image features and calculate similarity scores. The project also integrates with a **SQL Server** database to store and retrieve image metadata and supports hosting on **IIS**.

---

## **Features**
- Compare uploaded images with stored images based on similarity.
- Store image metadata in a SQL Server database.
- FastAPI endpoints for adding images and checking similarity.
- Easy integration with IIS for hosting.
- Uses **MobileNet** for feature extraction.

---

## **Requirements**

### **System Requirements**
- Python 3.10 or later
- IIS installed on the system
- SQL Server (any version with proper configuration)

### **Python Dependencies**
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```text
fastapi
uvicorn
pyodbc
keras
tensorflow
numpy
shutil
```

---

## **Directory Structure**
```
image-similarity/
├── main_override.py          # Main FastAPI application
├── model_util.py             # MobileNet feature extraction logic
├── web.config                # IIS hosting configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── (other project files...)
```

---

## **Endpoints**

### 1. **Check Similarity**
- **URL**: `/check_similarity/`
- **Method**: `POST`
- **Description**: Compare a target image with stored images.
- **Request Body**:
  - `product_code` (Form): The product code to match.
  - `target_image` (File): The image file to compare.
- **Response**:
  - `Matched` if a similar image is found.
  - `Did not match` otherwise.

### 2. **Add Image**
- **URL**: `/add_image/`
- **Method**: `POST`
- **Description**: Add a new image to the database for a specific product code.
- **Request Body**:
  - `product_code` (Form): The product code.
  - `image_file` (File): The image file to add.
- **Response**:
  - `Image added successfully` on success.

---

## **How It Works**
1. **Image Upload**: Upload images via the `/add_image/` endpoint.
2. **Database Storage**: Store the image file path and product code in a SQL Server database.
3. **Similarity Check**:
   - Extract features using **MobileNet**.
   - Calculate similarity using cosine similarity.
   - Match against stored images for the given product code.

---

## **Hosting on IIS**

### Prerequisites
1. **Install IIS**:
   - Go to **Control Panel > Programs > Turn Windows features on or off**.
   - Enable **Internet Information Services** and **Application Development Features** (CGI, ISAPI).
2. **Install `wfastcgi`**:
   ```bash
   pip install wfastcgi
   wfastcgi-enable
   ```

### Steps
1. **Place the Project**
   - Keep your project in a folder (e.g., `C:\Users\HP\Desktop\image-similarity`).

2. **Add a Website in IIS**
   - Open **IIS Manager**.
   - Right-click **Sites** → **Add Website**.
   - Configure:
     - **Site Name**: `ImageSimilarity`
     - **Physical Path**: Path to your project folder.
     - **Port**: `80` or another port.

3. **Add `web.config`**
   Place this file in the project folder:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <configuration>
     <system.webServer>
       <handlers>
         <add name="FastCGI" path="*" verb="*" modules="FastCgiModule" resourceType="Unspecified" />
       </handlers>
       <fastCgi>
         <application fullPath="C:\Python312\python.exe">
           <environmentVariables>
             <environmentVariable name="WSGI_SCRIPT" value="C:/Users/HP/Desktop/image-similarity/main_override.py" />
           </environmentVariables>
         </application>
       </fastCgi>
       <httpErrors errorMode="Detailed" />
     </system.webServer>
   </configuration>
   ```

4. **Set Permissions**
   - Right-click the project folder → **Properties > Security** → Add `IIS_IUSRS` with **Read & Execute** permissions.

5. **Restart IIS**
   ```bash
   iisreset
   ```

6. **Access the Application**
   - Open `http://localhost/` in your browser.
   - Swagger UI: `http://localhost/docs`.

---

## **Testing**
1. Use **Swagger UI** at `http://localhost/docs` to interact with the endpoints.
2. Test with tools like **Postman** or `curl`:
   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/check_similarity/' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'product_code=phone' \
     -F 'target_image=@path/to/your/image.jpg'
   ```

---

## **Troubleshooting**
- **422 Unprocessable Entity**:
  - Ensure you're using `multipart/form-data` for file uploads.
- **500 Internal Server Error**:
  - Check the `web.config` file and FastCGI configuration.
- **Database Errors**:
  - Verify the SQL Server connection string and permissions.

---

## **Future Improvements**
- Add support for more advanced image similarity models (e.g., Siamese Networks).
- Implement frontend for easier interaction.
- Add more robust error handling and logging.

---

## **License**
This project is licensed under the MIT License.
