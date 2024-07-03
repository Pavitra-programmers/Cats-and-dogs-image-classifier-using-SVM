import zipfile

with zipfile.ZipFile('datasets/test1.zip', 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall('test')
with zipfile.ZipFile('datasets/train.zip', 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall('train')
print(f"Extracted all files")
