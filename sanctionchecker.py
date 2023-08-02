from flask import Flask, request, redirect, url_for, flash, jsonify
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load('en_core_web_md')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)
        return data.to_json(orient='records')

@app.route('/analyze', methods=['POST'])
def analyze_file():
    json_data = request.get_json(force=True)
    name_column = json_data['NAME']
    description_column = json_data['INFO']
    
    data = pd.read_json(json_data['data'])
    sanctions = pd.read_csv('sanctionslist.csv')  # assuming this is the sanctions file

    name_matches = data[name_column][data[name_column].isin(sanctions['Name'])]
    
    # Vectorizing description and sanctioned names
    description_vectors = data[description_column].apply(lambda x: nlp(x).vector)
    sanction_vectors = sanctions['Name'].apply(lambda x: nlp(x).vector)

    # Calculating cosine similarity
    similarities = cosine_similarity(description_vectors.tolist(), sanction_vectors.tolist())

    # Let's say we consider descriptions with a similarity greater than 0.8 as a match
    match_indices = similarities > 0.8
    description_matches = data[description_column][match_indices.any(axis=1)]
    
    result = {"name_matches": name_matches.tolist(), "description_matches": description_matches.tolist()}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
