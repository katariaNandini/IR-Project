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
tfidf_matrix['Post ID'] = tfidf_matrix.index + 1  # Adjust the index to start from 1
tfidf_matrix['Text'] = tfidf_matrix['Post ID'].map(post_texts)

# Perform KMeans clustering
num_clusters = 5  # Choose the number of clusters based on previous analysis
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix.iloc[:, 1:-2])  # Exclude Post ID, Text for clustering

# Add the cluster labels to the DataFrame
tfidf_matrix['Cluster'] = clusters

# Save data to an Excel file
output_file = 'clustered_posts.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Save the main DataFrame
    tfidf_matrix.to_excel(writer, sheet_name='Clustered Posts', index=False)

    # Save a summary sheet with cluster top terms
    feature_names = tfidf_matrix.columns[1:-2]  # Exclude non-feature columns
    cluster_summary = []
    for i in range(num_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_terms_idx = cluster_center.argsort()[-10:][::-1]  # Get top 10 terms
        top_terms = [feature_names[j] for j in top_terms_idx]
        cluster_summary.append({"Cluster": i, "Top Terms": ", ".join(top_terms)})

    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_excel(writer, sheet_name='Cluster Summary', index=False)

print(f"Clustered posts saved to '{output_file}'")

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
