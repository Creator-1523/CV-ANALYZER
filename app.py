import io
import datetime
import time
import random
import base64
from pdfminer3.converter import TextConverter
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
from pyresparser import ResumeParser
from werkzeug.utils import secure_filename
import os
import re
import pandas as pd
from flask import Flask, render_template, request, redirect, flash, url_for
import pickle
import nltk
import spacy
nltk.download('stopwords')
spacy.load('en_core_web_sm')
ALLOWED_EXTENSIONS = set(['pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

tokenizer = pickle.load(open("models/tfidf.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


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
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(
        resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()
    # close open handles
    converter.close()
    fake_file_handle.close()
    return text


app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    # if request.method == "POST":
    #     pass
    return render_template("index.html")


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        savefile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(savefile)

        print('upload_image filename: ' + filename)

        print("Model file loaded")
        # filename=r"static\Charmi_Resume_.pdf"
        resume_data = ResumeParser(savefile).get_extracted_data()
        if resume_data:
            resume_text = pdf_reader(
                (os.path.join(app.config['UPLOAD_FOLDER'], filename)))
            resume_score = 0
            if "EDUCATION" in resume_text:
                resume_score = resume_score+20
            else:
                resume_score = resume_score-20

        print(resume_data['email'])
        print()

        print("Predicting")
        # Clean the input resume
        cleaned_resume = cleanResume(pdf_reader(
            os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        # print(cleaned_resume)
        input_features = tokenizer.transform([cleaned_resume])
        prediction_id = model.predict(input_features)[0]
        print(prediction_id)
        category_name = category_mapping.get(prediction_id, "Unknown")
        print(category_name)
        return render_template('resume.html', result=category_name, cleaned_resume=cleaned_resume, resume_score=resume_score, filename=os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # return None


if __name__ == "__main__":
    app.run(debug=True)
