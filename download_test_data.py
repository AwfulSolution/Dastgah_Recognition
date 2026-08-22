import os
import requests

def download_file(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    r = requests.get(url, allow_redirects=True)
    with open(os.path.join(folder, filename), 'wb') as f:
        f.write(r.content)

# We will just generate some tone data for fast testing.
