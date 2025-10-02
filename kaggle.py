import kagglehub

# Download latest version
path = kagglehub.dataset_download("dansbecker/powerlifting-database")

print("Path to dataset files:", path)