import subprocess
import os

# List of Python scripts to run in order
scripts = [
    # '1reddit_api.py',             # Runs first and saves content in reddit_posts.json
    '2preprocess_reddit_data.py', # Preprocesses the Reddit data
    '3tf_idf.py',                 # Generates TF-IDF vectors
    '4topic_modeling.py',         # Runs topic modeling
    '5clustering.py',             # Runs clustering
    '6generate_cluster_with_lda.py', # Generates clusters with LDA
    '7classification.py',
    '8custmo.py'                  # Custom script
]

# Input for reddit_api.py
reddit_api_input = "technology\n50\nPC\n"  # Example: subreddit='technology', posts=50, query='PC'

# Run each script sequentially
for script in scripts:
    print(f"Running {script}...")
    
    # Check if the script exists
    if os.path.exists(script):
        print(f"Found {script}... Running it now.")
        
        try:
            if script == 'reddit_api.py':
                # Use Popen to handle input properly
                process = subprocess.Popen(
                    ['python', script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=reddit_api_input)
            else:
                # Run other scripts normally
                process = subprocess.run(
                    ['python', script],
                    capture_output=True,
                    text=True
                )
                stdout, stderr = process.stdout, process.stderr
            
            # Print script output
            print(stdout)
            
            # Print errors, if any
            if stderr:
                print(f"Error in {script}:")
                print(stderr)
            
            print(f"Finished running {script}\n")
        except Exception as e:
            print(f"An error occurred while running {script}: {e}")
    else:
        print(f"{script} does not exist in the current directory. Please check the script name or path.\n")
