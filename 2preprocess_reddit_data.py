import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Load data from reddit_posts.json
with open("reddit_posts.json", "r") as f:
    data = json.load(f)

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Apply preprocessing to each post's content
preprocessed_data = []
for i, post in enumerate(data, start=1):  # Start enumeration from 1
    content = post['title'] + " " + post['body']  # Concatenate title and body
    cleaned_content = preprocess_text(content)    # Preprocess text
    preprocessed_data.append({
        "post_id": i,  # Assign a sequential post_id starting from 1
        "cleaned_content": cleaned_content,
        "original_data": post
    })

# Save preprocessed data to a new JSON file
with open("preprocessed_reddit_posts.json", "w") as f:
    json.dump(preprocessed_data, f, indent=4)

print("Preprocessed data with post_id saved to 'preprocessed_reddit_posts.json'")
