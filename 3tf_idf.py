import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data from JSON file
with open('preprocessed_reddit_posts.json', 'r') as f:
    data = json.load(f)

# Extract the 'cleaned_content' field from each post
cleaned_contents = [post['cleaned_content'] for post in data]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(cleaned_contents)

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Save the TF-IDF matrix to a CSV file
tfidf_df.to_csv('tfidf_matrix.csv', index=False)
print("TF-IDF matrix saved to 'tfidf_matrix.csv'.")
