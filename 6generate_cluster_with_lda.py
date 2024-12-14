import csv
import json

# Load LDA topics into a dictionary based on cluster ID
lda_topics = {}
with open('lda_topics.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            cluster_id = int(float(row['LDA_Topic']))  # Convert to float first, then to int
            lda_topics[cluster_id] = row['LDA_Topic_Name']
        except ValueError:
            print(f"Skipping row due to conversion error: {row}")

# Load clustered posts data
clustered_posts = []
with open('clustered_posts.csv', mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header row
    for row in reader:
        try:
            post_id = int(float(row[0]))  # Convert to float first, then to int
            text = row[1]
            cluster_id = int(float(row[-1]))  # Assuming the last column is the cluster ID
            clustered_posts.append((post_id, text, cluster_id))
        except ValueError:
            print(f"Skipping row due to conversion error: {row}")

# Create combined output file
with open('clustered_posts_with_lda.csv', mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Post ID', 'Text', 'LDA_Topic', 'LDA_Topic_Name'])

    for post_id, text, cluster_id in clustered_posts:
        lda_topic_name = lda_topics.get(cluster_id, "Unknown")  # Default to "Unknown" if cluster_id not found
        writer.writerow([post_id, text, cluster_id, lda_topic_name])

print("Combined file saved as clustered_posts_with_lda.csv")
