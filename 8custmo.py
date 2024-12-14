import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# === Configuration ===
input_file = "clustered_posts.csv"  # Input CSV file with pre-clustered data (from the first script)
ground_truth = None  # Replace with ground truth labels if available

# === Load Data ===
df = pd.read_csv(input_file)
df['Text'] = df['Text'].fillna('')  # Ensure non-empty text

# === 1. Internal Evaluation ===
# Silhouette Score (Optional as ground truth is not available here)
silhouette_avg = silhouette_score(df.iloc[:, 2:-1], df['Cluster'])  # Excluding 'Post ID', 'Text', and cluster columns
print(f"Silhouette Score: {silhouette_avg}")

# === 2. External Evaluation (if ground truth available) ===
if ground_truth:
    ari = adjusted_rand_score(ground_truth, df['Cluster'])
    nmi = normalized_mutual_info_score(ground_truth, df['Cluster'])
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")

# === 3. Analyze Cluster Content ===
clustered_data = df.groupby('Cluster')['Text'].apply(list).to_dict()

# Analyze each cluster
for cluster_id, texts in clustered_data.items():
    # Word Frequencies
    words = " ".join(texts).split()
    word_freq = Counter(words)
    print(f"Cluster {cluster_id} - Top Words: {word_freq.most_common(5)}")

    # Sentiment Analysis
    sentiments = [TextBlob(text).sentiment.polarity for text in texts if text.strip()]
    avg_sentiment = np.mean(sentiments) if sentiments else None
    print(f"Cluster {cluster_id} - Average Sentiment: {avg_sentiment}")

# === 4. Visualize Clusters ===

# 4.a Word Clouds for Each Cluster
for cluster_id, texts in clustered_data.items():
    if not texts:
        print(f"Cluster {cluster_id} has no valid text for word cloud.")
        continue
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure()
    plt.title(f"Cluster {cluster_id} - Word Cloud")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# 4.b PCA Visualization for Clusters
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(df.iloc[:, 2:-1].values)  # Use the features excluding 'Post ID', 'Text', and 'Cluster'
plt.figure(figsize=(8, 6))
for cluster_id in range(df['Cluster'].nunique()):
    plt.scatter(
        X_reduced[df['Cluster'] == cluster_id, 0],
        X_reduced[df['Cluster'] == cluster_id, 1],
        label=f"Cluster {cluster_id}"
    )
plt.title("Cluster Visualization (PCA)")
plt.legend()
plt.show()

# === End of Script ===
