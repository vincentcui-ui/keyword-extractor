import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import PyPDF2
import io

nltk.download('punkt')
nltk.download('stopwords')

sanctions_list = []

with open('/Users/vincent_cui/Keyword-Extracter/sanctionlist.csv', 'r') as file:
    reader = csv.reader(file)

    next(reader)  # skip header row

    for row in reader:
        print(row)  # print the row
        sanctions_list.append(row[2])  # append name from Column B (indexed as 1)