import praw
import prawcore  # Import prawcore to handle exceptions
import json

# Set up Reddit API credentials
reddit = praw.Reddit(
    client_id='wjwQOaL_falBQJofigVgSQ',       # Replace with your Client ID
    client_secret='Ys2hBW1bD4w6WxxKa1psX1B9GVphdw',  # Replace with your Client Secret
    user_agent='postclassifier'
)

def fetch_reddit_posts(subreddit_name, limit=10):
    while True:
        try:
            # Replace spaces with underscores in subreddit names to match Reddit's format
            subreddit_name = subreddit_name.replace(" ", "")
            subreddit = reddit.subreddit(subreddit_name)
            data = []
            for submission in subreddit.new(limit=limit):
                post = {
                    "title": submission.title,
                    "body": submission.selftext,
                    "author": str(submission.author),
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "url": submission.url,
                    "is_video": submission.is_video
                }
                data.append(post)
            return data
        except prawcore.exceptions.NotFound:
            print(f"\nError: Subreddit '{subreddit_name}' not found. Please check the subreddit name.")
            return []

# Take input for subreddit name and number of posts
subreddit_name = input("Enter the subreddit name: ")
limit = int(input("Enter the number of posts to fetch: "))

# Fetch the posts
data = fetch_reddit_posts(subreddit_name, limit)

# Save fetched data to a JSON file
if data:
    with open("reddit_posts.json", "w") as f:
        json.dump(data, f, indent=4)
    print("\nData saved to 'reddit_posts.json'")
else:
    print("\nNo data fetched. Exiting.")
