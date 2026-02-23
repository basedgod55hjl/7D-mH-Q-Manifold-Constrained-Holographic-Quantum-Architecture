import os, hashlib, json

def hash_file(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(root):
    hashes = {}
    duplicates = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            try:
                h = hash_file(filepath)
                if h in hashes:
                    duplicates.append((filepath, hashes[h]))
                else:
                    hashes[h] = filepath
            except Exception:
                continue
    return duplicates

def delete_duplicates(root):
    dupes = find_duplicates(root)
    for dup, original in dupes:
        try:
            os.remove(dup)
            print(f"Deleted duplicate: {dup} (original: {original})")
        except Exception as e:
            print(f"Failed to delete {dup}: {e}")

if __name__ == "__main__":
    root = r"C:/Users/BASEDGOD/Desktop/SYSTEM CONFIG/11_REPO_VAULT/7D-mH-Q-Manifold-Constrained-Holographic-Quantum-Architecture"
    delete_duplicates(root)
