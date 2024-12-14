import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load Data
with open('preprocessed_reddit_posts.json', 'r') as f:
    data = json.load(f)

# Extract cleaned content for TF-IDF
cleaned_contents = [post['cleaned_content'] for post in data]
post_ids = [post['post_id'] for post in data]

# Step 2: Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(cleaned_contents)

# Step 3: Clustering with KMeans
optimal_clusters = 5  # Set based on elbow or silhouette method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(X_tfidf)

# Step 4: Add True Labels (for evaluation)
# In practice, you should have true labels for evaluation.
# Here, we assume the true labels are included in your dataset for demonstration purposes.
# Add a column with true labels to simulate evaluation. Replace with actual labels.
true_labels = [post.get('true_label', None) for post in data]  # Replace None with actual logic to extract labels

# Step 5: Evaluation
if None not in true_labels:  # Proceed only if true labels are available
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

# Step 6: Save Clustering Results
clustered_data = pd.DataFrame({
    "Post ID": post_ids,
    "Text": cleaned_contents,
    "Cluster": predicted_labels
})
clustered_data.to_csv('clustered_posts_with_labels.csv', index=False)
print("Clustered posts with predicted labels saved to 'clustered_posts_with_labels.csv'.")

# Step 7: Optional Visualization with PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predicted_labels, cmap='viridis', alpha=0.6)
plt.title('PCA of TF-IDF data with KMeans clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
