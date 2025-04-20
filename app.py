import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline

# Function to extract text from image
# Function to extract text from image using EasyOCR
def extract_text(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path, detail=0)
    text = " ".join(results)
    return text


# Function to generate questions from extracted text
def generate_questions(text):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    result = question_generator(text, num_return_sequences=5, num_beams=5)
    questions = [r['generated_text'] for r in result]
    return questions

# Streamlit UI
st.title("AI Question Generator")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Extract text from the uploaded image
    extracted_text = extract_text("temp_image.jpg")
    st.write("📄 Extracted Text:")
    st.write(extracted_text)
    
    # Generate and display AI-generated questions
    st.write("❓ AI Generated Questions:")
    questions = generate_questions(extracted_text)
    for q in questions:
        st.write(f"- {q}")

