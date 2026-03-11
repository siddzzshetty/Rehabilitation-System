import streamlit as st
import os
import io
import cv2
import numpy as np
import fitz  # PyMuPDF for text extraction
from google.cloud import vision
from PIL import Image
from fuzzywuzzy import process
from pdf2image import convert_from_bytes
import tempfile
Image.MAX_IMAGE_PIXELS = None  

# Set up authentication using the JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "carbon-pride-453005-g3-2b983e32ddd4.json"

# Predefined exercise keywords
exercise_keywords = ["squat", "pushup", "jumping jack", "sit up", "pull-up"]

# Initialize session state variables if they don't exist
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "prescribed_exercises" not in st.session_state:
    st.session_state["prescribed_exercises"] = []

def preprocess_image(image):
    try:
        # Convert to numpy array
        image = np.array(image)

        # Resize if image is too large
        max_size = 4000  # Set a reasonable max size (width/height)
        if image.shape[0] > max_size or image.shape[1] > max_size:
            scale_factor = max_size / max(image.shape[:2])
            new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Check if the image is already in grayscale (1 channel)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = cv2.GaussianBlur(image, (5, 5), 0)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        temp_image_path = "temp_processed.jpg"
        cv2.imwrite(temp_image_path, image)
        return temp_image_path
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

def extract_text_google_vision(image_path):
    try:
        client = vision.ImageAnnotatorClient()
        with io.open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        extracted_text = response.full_text_annotation.text if response.full_text_annotation.text else ""
        
        os.remove(image_path)  # Cleanup temp file
        return extracted_text
    except Exception as e:
        st.error(f"Error in Google Vision API: {e}")
        return ""

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""

    for page in doc:
        extracted_text += page.get_text("text")

    if extracted_text.strip():
        return extracted_text  # If text is found, return it

    images = convert_from_bytes(pdf_bytes, dpi=300)  # Convert PDF to images
    text_from_images = []
    
    for i, img in enumerate(images):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image_path = temp_file.name
                img.save(image_path, "JPEG")

            if not os.path.exists(image_path):  # Check if the file is actually created
                st.error(f"Error: File {image_path} not found after saving.")
                continue
            
            text = extract_text_google_vision(image_path)
            text_from_images.append(text)
        
        except Exception as e:
            st.error(f"Error processing image {image_path}: {e}")
        
        finally:
            if os.path.exists(image_path):  
                os.remove(image_path)  # Remove file only if it exists
    
    return " ".join(text_from_images)

def extract_prescribed_exercises(text):
    try:
        exercises = []
        lines = text.splitlines()
        for line in lines:
            match, score = process.extractOne(line.lower(), exercise_keywords)
            if score > 70:
                exercises.append(match)
        return exercises
    except Exception as e:
        st.error(f"Error extracting exercises: {e}")
        return []

st.title("Welcome! Your Rehabilitation Assistant is here to guide you!")

if not st.session_state["logged_in"]:
    with st.form("login_form"):
        email = st.text_input("Email", value="assistant@gmail.com")
        password = st.text_input("Password", value="pass@123", type="password")
        uploaded_file = st.file_uploader("Upload Prescription (Image/PDF)", type=["jpg", "jpeg", "png", "pdf"])
        
        submitted = st.form_submit_button("Login")
        
        if submitted and uploaded_file is not None:
            extracted_text = ""
            
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file.read())
            else:
                image = Image.open(uploaded_file)
                processed_path = preprocess_image(image)
                if processed_path:
                    extracted_text = extract_text_google_vision(processed_path)
            
            if extracted_text.strip():
                st.session_state["prescribed_exercises"] = extract_prescribed_exercises(extracted_text)
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("No prescribed exercises detected. Try again with a clearer document.")
else:
    st.success("Successfully logged in!")
    
    st.subheader("Identified Prescribed Exercises:")
    if st.session_state["prescribed_exercises"]:
        for exercise in st.session_state["prescribed_exercises"]:
            if st.button(exercise.title()):
                st.session_state.selected_exercise = exercise.lower()  # Save exercise
                st.session_state.camera_started = True  # Optional, to auto-start camera
                st.switch_page("pages/final1.py")  # Switch to camera feed page
    else:
        st.write("No prescribed exercises detected.")
    
    st.subheader("Upload Another Prescription")
    new_uploaded_file = st.file_uploader("Upload Another Prescription (Image/PDF)", type=["jpg", "jpeg", "png", "pdf"])
    if new_uploaded_file is not None:
        extracted_text = ""
        if new_uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(new_uploaded_file.read())
        else:
            image = Image.open(new_uploaded_file)
            processed_path = preprocess_image(image)
            if processed_path:
                extracted_text = extract_text_google_vision(processed_path)
        
        if extracted_text.strip():
            st.session_state["prescribed_exercises"] = extract_prescribed_exercises(extracted_text)
            st.rerun()
        else:
            st.error("No prescribed exercises detected. Try again with a clearer document.")