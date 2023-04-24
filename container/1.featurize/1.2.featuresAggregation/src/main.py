import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from scipy import sparse
from google.cloud import storage

def exec_featurize():
    # Env
    print('- Start -')
    BUCKET=os.getenv('BUCKET')
    
    # Download inputs
    client= storage.Client()
    bucket = client.get_bucket(BUCKET)
    blobs = bucket.list_blobs(prefix="featurized/tmp/", delimiter="/")

    X=[]
    c=0
    print(" Feature List : [ ")
    for blob in blobs:
        blobtmp=blob.name[15:]
        print(blobtmp)
        blob.download_to_filename(blobtmp)
        t=sparse.load_npz(blobtmp)
        X.append(t)
        c += 1
    print("]")
    print("- Feature Count : " + str(c) + " -")
    X = vstack(X) # Convert X from a list of csr_matrix to just a csr_matrix

    sparse.save_npz("X.npz", X)
    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("featurized/X.npz")
    blob.upload_from_filename("./X.npz")

    print(" - Complete - ")

if __name__ == '__main__':
    exec_featurize()