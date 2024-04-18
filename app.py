from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models.coherencemodel import CoherenceModel
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Initialize Flask app
app = Flask(__name__)

# Load necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data and preprocess it
data = pd.read_csv("training_articles_csv_file.csv", encoding='latin1')

# Preprocess text function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    # Join tokens to form preprocessed text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

vectorizer =TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
x = vectorizer.fit_transform(data['article'].values)
lda = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42)
lda.fit(x)
topic_modelling = lda.transform(x)

topic_labels = np.argmax(topic_modelling, axis=1)
data['topic_labels'] = topic_labels
@app.route('/')
def index():
    
    # HTML content with embedded JavaScript
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Modeling Demo</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <h1>Topic Modeling Demo</h1>
        <form id="topicForm">
            <label for="textInput">Enter text:</label><br>
            <input type="text" id="textInput" name="text"><br>
            <button type="submit">Submit</button>
        </form>
        <div id="output"></div>
        <script>
            $(document).ready(function() {
                $('#topicForm').submit(function(event) {
                    event.preventDefault();
                    var text = $('#textInput').val();
                    $.ajax({
                        url: '/topic_modeling',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ text: text }),
                        success: function(response) {
                            $('#output').text(JSON.stringify(response));
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """
# Define route for topic modeling
@app.route('/topic_modeling', methods=['POST'])
def topic_modeling():
    
    # Get text data from request
    requested_data = request.get_json()
    new_article_text=requested_data['text']

    # Preprocess the text
    preprocessed_text = preprocess_text(new_article_text)

    # Transform preprocessed text into vector space
    
    tfidf_new_article = vectorizer.transform([preprocessed_text])

    # Transform vectorized text into topic space
    
    topic_modelling_new_article = lda.transform(tfidf_new_article)

    # Find dominant topic
    dominant_topic_index = np.argmax(topic_modelling_new_article)
    dominant_topic = dominant_topic_index

    # Filter rows based on dominant topic
    filtered_rows = data[data['topic_labels'] == dominant_topic]

    # Prepare response
    suggested_tags = filtered_rows['topic'].tolist()
    if not filtered_rows.empty:
        return jsonify({'suggested_tags': suggested_tags})
    else:
        return jsonify({'message': 'No matching rows found'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

