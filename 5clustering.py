import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load TF-IDF matrix from CSV
tfidf_matrix = pd.read_csv('tfidf_matrix.csv')

# Load the preprocessed Reddit posts JSON
with open('preprocessed_reddit_posts.json', 'r', encoding='utf-8') as f:
    posts_data = json.load(f)

# Create a dictionary to map post_id to text
post_texts = {post['post_id']: post['cleaned_content'] for post in posts_data}

# Ensure post_id exists in the tfidf_matrix for alignment
# Adjusting post_id to start from 1 instead of 0
tfidf_matrix['Post ID'] = tfidf_matrix.index + 1  # Adjust the index to start from 1
tfidf_matrix['Text'] = tfidf_matrix['Post ID'].map(post_texts)

# Range of cluster numbers to test (Elbow Method)
num_clusters_range = range(1, 11)  # Try 1 to 10 clusters
inertia = []
sil_scores = []

# Use KMeans with different cluster numbers and evaluate clustering quality
for n in num_clusters_range:
    kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)  # Increased n_init for stability
    kmeans.fit(tfidf_matrix.iloc[:, 1:-2])  # Excluding Post ID, Text, and Cluster columns for clustering
    
    inertia.append(kmeans.inertia_)
    
    # Ensure more than one cluster is formed before calculating silhouette score
    unique_labels = set(kmeans.labels_)
    if len(unique_labels) > 1 and len(unique_labels) <= len(tfidf_matrix) - 1:  # Check for valid cluster number
        sil_scores.append(silhouette_score(tfidf_matrix.iloc[:, 1:-2], kmeans.labels_))
    else:
        sil_scores.append(-1)  # Silhouette score isn't valid for a single cluster or too many clusters

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(num_clusters_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal K')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(num_clusters_range, sil_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')

plt.tight_layout()
plt.show()

# Based on the elbow and silhouette scores, choose an optimal cluster number (e.g., 5)
num_clusters = 5  # Adjust this based on analysis from above plots

# Initialize KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix.iloc[:, 1:-2])  # Exclude Post ID, Text for clustering

# Add the cluster labels to the DataFrame
tfidf_matrix['Cluster'] = clusters

# Get the top terms for each cluster
feature_names = tfidf_matrix.columns[1:-2]  # Excluding 'Post ID' and 'Text'
top_terms_per_cluster = {}
for i in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_idx = cluster_center.argsort()[-10:][::-1]  # Get the top 10 terms
    top_terms = [feature_names[j] for j in top_terms_idx]
    top_terms_per_cluster[i] = top_terms
    print(f"Cluster {i}: {', '.join(top_terms)}")

# Save the final output to CSV
# We need to select only the Post ID, Text columns, and then the remaining weight columns plus the cluster
tfidf_matrix.to_csv('clustered_posts.csv', columns=['Post ID', 'Text'] + list(tfidf_matrix.columns[1:-2]) + ['Cluster'], index=False)

print("Clustered posts saved to 'clustered_posts.csv'")

# Optional: Dimensionality Reduction for Visualization (PCA)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.iloc[:, 1:-2])

# Plotting the PCA-transformed data with clusters
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('PCA of TF-IDF data with KMeans clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

