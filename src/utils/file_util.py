import os
def find_project_root(current_path):
    markers = ['.git', 'README.md', '.gitignore']

    while current_path != os.path.dirname(current_path):
        if any(os.path.exists(os.path.join(current_path, marker)) for marker in markers):
            return current_path
        current_path = os.path.dirname(current_path)
    return None


import random

def split_dict(data, test_ratio=0.1):
    keys = list(data.keys())
    test_size = int(len(keys) * test_ratio)
    test_keys = random.sample(keys, test_size)
    train_keys = [key for key in keys if key not in test_keys]
    train_data = {key: data[key] for key in train_keys}
    test_data = {key: data[key] for key in test_keys}

    return train_data, test_data