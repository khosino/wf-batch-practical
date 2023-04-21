import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from scipy import sparse
from google.cloud import storage

def exec_featurize():
    # Load environment variables
    BUCKET=os.getenv('BUCKET')
    INPUT_BUCKET=os.getenv('INPUT_BUCKET')
    num_tasks_featurizer=int(os.environ.get('NUM_TASKS_FEATURIZER'))
    BATCH_TASK_COUNT=int(os.environ.get('BATCH_TASK_COUNT'))
    BATCH_TASK_INDEX=int(os.environ.get('BATCH_TASK_INDEX'))

    # Display the Number of features and Index in Parallel processing
    print("Number of Features: " + str(BATCH_TASK_COUNT))
    print("Index: " + str(BATCH_TASK_INDEX+1))

    # Download input files from Input bucket and load to variable
    client= storage.Client()
    bucket = client.bucket(INPUT_BUCKET)
    blob = bucket.blob("input/X_raw")
    blob.download_to_filename("./X_raw")

    f = open("./X_raw","rb")
    X_raw = pickle.load(f)

    # Featurize
    ## 
    batch_size = len(X_raw) // num_tasks_featurizer + 1
    vectorizer = TfidfVectorizer(max_features=300)
    vectorizer.fit(X_raw)
    ## Run with the allocated task index
    Xi_name = "X"+str(BATCH_TASK_INDEX)+".npz"
    Xi=vectorizer.transform(X_raw[BATCH_TASK_INDEX * batch_size:(BATCH_TASK_INDEX + 1) * batch_size])

    sparse.save_npz(Xi_name, Xi)
    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("featurized/tmp/"+Xi_name)
    blob.upload_from_filename("./"+Xi_name)

    print("- Complete -")

if __name__ == '__main__':
    exec_featurize()