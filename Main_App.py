import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')

# Load the classifier and TF-IDF model
clf = pickle.load(open(r"C:\Users\Dell\OneDrive\Desktop\Project\clf.pkl", 'rb'))
tfidfd = pickle.load(open(r"C:\Users\Dell\OneDrive\Desktop\Project\tfidf.pkl", 'rb'))

# Clean resume text function
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

# Extract text from PDF files
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Streamlit web app
def main():
    st.title("Automatic CV Evaluator")

    uploaded_file = st.file_uploader("Upload a resume (PDF or TXT) to predict the job category.",type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'application/pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        # Clean the resume text
        cleaned_resume = clean_resume(resume_text)

        # Transform the input text using the TF-IDF model
        input_features = tfidfd.transform([cleaned_resume])

        # Predict the category using the loaded classifier
        try:
            prediction_id = clf.predict(input_features)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display the predicted category
        st.success(f"Predicted Category: {category_name}")

# Python entry point
if __name__ == "__main__":
    main()
