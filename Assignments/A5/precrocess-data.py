import os
import requests


def download_file(url, folder):
    local_filename = url.split('/')[-1]
    path = os.path.join(folder, local_filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            # the response from the server will be processed in pieces of 8192 bytes (or 8 KB) at a time
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path


def clean_and_merge_files(paths, output_folder):
    all_names = []
    for path in paths:
        count = 0
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                name = line.strip().split('\t')[0]
                all_names.append(name)
                count += 1
        print(f"Total names in {os.path.basename(path)}: {count}")

    with open(os.path.join(output_folder, 'all_names.txt'), 'w') as file:
        for name in all_names:
            file.write(name + '\n')
    print(f"Total names in merged file: {len(all_names)}")


def download_and_merge_names():
    base_url = "https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/"
    files = ["female.txt", "male.txt", "pet.txt", "other/names.txt"]
    paths = []

    original_folder = 'dataset/original'
    merged_folder = 'dataset/merged'
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(merged_folder, exist_ok=True)

    for file in files:
        print(f"Downloading {file}...")
        file_path = download_file(base_url + file, original_folder)
        paths.append(file_path)

    clean_and_merge_files(paths, merged_folder)
    print("Download and merging complete.")


if __name__ == '__main__':
    print("Downloading dataset from https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/")
    download_and_merge_names()
