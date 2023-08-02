from flask import Flask, request, render_template
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import pandas as pd

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

def process_sanctions_list(file_path):
    sanctions_list = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            names = row["NAME"].split()
            names = [name for name in names if name not in stop_words]
            sanctions_list.extend(names)
            
            additional_info = row["INFO"]
            if additional_info:
                for info in additional_info.split(';'):
                    sanctions_list.append(info.strip())
    sanctions_list = [name.lower() for name in sanctions_list]
    sanctions_list = list(set(sanctions_list))
    return sanctions_list

def find_keywords_in_text(text, keywords):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    
    matches = []
    for word in words:
        for keyword in keywords:
            if fuzz.ratio(word, keyword) > 80:
                matches.append((word, keyword))
    return matches

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        data = pd.read_csv(file)
        text = data['Name'].astype(str) + ' ' + data['Description'].astype(str)
        text = ' '.join(text)
        
        sanctions_list = process_sanctions_list('sanctionlist.csv')

        sentences = nltk.sent_tokenize(text)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        words = vectorizer.get_feature_names_out()
        sums = np.array(tfidf_matrix.sum(axis=0))[0]
        word_scores = {word: sums[idx] for word, idx in zip(words, range(len(words)))}

        sanctions_list = [word for word in sanctions_list if word in word_scores and word_scores[word] > 0.01]

        matches = find_keywords_in_text(text, sanctions_list)

        return render_template('results.html', matches=matches)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
