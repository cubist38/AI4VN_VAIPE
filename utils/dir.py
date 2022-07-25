import os

def create_directory(directory):
    dirs = directory.split('/')
    path = ''
    for folder in dirs:
        path = os.path.join(path, folder)
        if not os.path.exists(path):
            os.makedirs(path)