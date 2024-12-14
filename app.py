from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_reddit_posts', methods=['POST'])
def fetch_reddit_posts():
    # Get form data
    subreddit_name = request.form['subreddit_name']
    limit = int(request.form['limit'])

    # Write the inputs to a JSON file for later use
    query_info = {
        "subreddit_name": subreddit_name,
        "limit": limit
    }
    
    with open("query_input.json", "w") as f:
        json.dump(query_info, f, indent=4)

    # Run the reddit_api.py script to fetch posts
    result = subprocess.run(['python', 'reddit_api.py'], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # After fetching posts, run the other scripts
    scripts = [
        'preprocess_reddit_data.py',
        'tf_idf.py',
        'topic_modeling.py',
        'clustering.py',
        'generate_cluster_with_lda.py',
        'classification.py',
        'custmo.py'
    ]
    
    for script in scripts:
        print(f"Running {script}...")
        if os.path.exists(script):
            result = subprocess.run(['python', script], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Error in {script}:")
                print(result.stderr)
        else:
            print(f"{script} does not exist in the current directory.\n")

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
