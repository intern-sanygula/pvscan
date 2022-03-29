from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import json
import pandas as pd
import os
import pickle
import string
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer
import pytesseract
from pdf2image import convert_from_path
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pdf2image import convert_from_path
import glob
import pytesseract

savedmodel = pickle.load(open('nb.pkl','rb'))
tfidfconverter = pickle.load(open('tf01.pkl', 'rb'))
labelencoder = pickle.load(open('le.pkl', 'rb'))
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = os.getcwd()

# initialzing flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extractTextFromPDF(pdf_location):
    text = ""
    pdfs = glob.glob(pdf_location)
    for pdf_path in pdfs:
        pages = convert_from_path(pdf_path, 500, poppler_path = r'C:\Program Files\poppler-0.68.0\bin'

        for pageNum,imgBlob in enumerate(pages):
            text_per_page = pytesseract.image_to_string(imgBlob, lang='eng')
            text += text_per_page + " "
    
    return (" ".join(text.split())).strip()

#home function
@app.route('/')

def home():
    return render_template('index.html')

def predict_topic(text):

    df = pd.DataFrame({"Data" : [text]})
    # Feature engineering to get the data in right format
    df['Data'] = df['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversion
    df['Data'] = df['Data'].str.replace('[^\w\s]','') # getting rid of special characters
    df['Data'] = df['Data'].str.replace('\d+', '') # removing numeric values from between the words
    df['Data'] = df['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers
    stop = stopwords.words('english')
    df['Data'] = df['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop and len(x) > 2 and len(x) < 15 and x.isalnum())) #removing stop words
    stemmer = WordNetLemmatizer()
    df['Data'] = [stemmer.lemmatize(word) for word in df['Data']]

    #BOW model
    
    inputs = tfidfconverter.transform(df['Data']).toarray()

    output_category = savedmodel.predict(inputs)
    output_category = (labelencoder.inverse_transform((output_category)))
    #Comment the next line if you are testing word2vec model as it doesn't require transformation
    return output_category[0]

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        file = request.files['file']

    try:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text_string = extractTextFromPDF(filename)
        category = predict_topic(text_string)
        old_name = filename
        new_name = category + ".pdf"
        os.rename(old_name, new_name)
        return render_template('index.html', category = category)
    except:
        return render_template('index.html')
        
    
if __name__ == "__main__":
    app.run(debug=True)
