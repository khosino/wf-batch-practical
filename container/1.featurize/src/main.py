import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from google.cloud import storage

def exec_featurize():
    # Env
    print('Start')
    BUCKET=os.getenv('BUCKET')
    num_tasks_featurizer=int(os.environ.get('num_tasks_featurizer'))

    # Download inputs
    print(BUCKET)
    print(num_tasks_featurizer)
    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("input/X_raw")
    blob.download_to_filename("./X_raw")

    f = open("./X_raw","rb")
    X_raw = pickle.load(f)

    # Featurize
    batch_size = len(X_raw) // num_tasks_featurizer + 1
    vectorizer = TfidfVectorizer(max_features=300)
    vectorizer.fit(X_raw)
    X = []
    for i in range(10):
        X.append(vectorizer.transform(X_raw[i * batch_size:(i + 1) * batch_size]))
    X = vstack(X)  # Convert X from a list of csr_matrix to just a csr_matrix

    # Upload to bucket
    f = open('X', 'wb')
    pickle.dump(X, f)

    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("featurized/X")
    blob.upload_from_filename("./X")

    print('complete')

if __name__ == '__main__':
    exec_featurize()