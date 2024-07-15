import os
# Set the environment variable within the script if it's not detected
if "DATA_TRAIN" not in os.environ:
    os.environ["DATA_TRAIN"] = "/Users/noymachluf/Desktop/pseudo-sr/dataset/train"

gh_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
print("Path to HIGH directory:", gh_folder)
# Build the path to the HIGH directory
gh_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")

# Print the path to ensure it's correct
print("Path to HIGH directory:", gh_folder)

# Optionally, list a few files in the directory to confirm it's correct
if os.path.exists(gh_folder):
    # List the first few files in the directory
    files = os.listdir(gh_folder)
    print("Some files in HIGH directory:", files[:5])  # Print first 5 files
else:
    print("Directory does not exist:", gh_folder)