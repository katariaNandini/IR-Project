import json
import csv
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load preprocessed data from JSON
with open('preprocessed_reddit_posts.json', 'r') as f:
    data = json.load(f)

# Extract cleaned content and assign IDs for tracking
documents = [(idx, post["cleaned_content"]) for idx, post in enumerate(data)]

# Tokenize the cleaned content
tokenized_texts = [doc[1].split() for doc in documents]

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# Define the number of topics and create LDA model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Function to generate and save LDA topics as dictionary with topic names
def get_lda_topics(lda_model, num_topics):
    lda_topics = {}
    for idx, topic in lda_model.print_topics(num_words=5):
        topic_words = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]
        lda_topics[idx] = ", ".join(topic_words)
    return lda_topics

# Get LDA topics as dictionary
lda_topics = get_lda_topics(lda_model, num_topics)

# Save LDA topics and associated post information to a CSV file
output_file = 'lda_topics.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Post ID', 'Text', 'LDA_Topic', 'LDA_Topic_Name'])

    # Assign each post to its most probable topic
    for idx, bow in enumerate(corpus):
        post_id, text = documents[idx]
        topic_distribution = lda_model.get_document_topics(bow)
        most_probable_topic = max(topic_distribution, key=lambda x: x[1])[0]
        topic_name = lda_topics[most_probable_topic]
        
        # Write each post's details to the CSV
        writer.writerow([post_id, text, most_probable_topic, topic_name])

print(f"Posts with LDA topics have been saved to {output_file}")

# Plot word clouds for each topic
def plot_word_cloud(lda_model, num_topics):
    for t in range(num_topics):
        plt.figure(figsize=(8, 6))
        plt.imshow(WordCloud(width=800, height=400, background_color='white').fit_words(dict(lda_model.show_topic(t, 20))))
        plt.axis("off")
        plt.title(f"Topic {t + 1}: {lda_topics[t]}", fontsize=16)
        plt.show()

plot_word_cloud(lda_model, num_topics)
