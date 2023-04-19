import numpy as np
import os
import pickle
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage

def exec_train_aggregate():
    # Env
    print('- Start -')
    BUCKET=os.getenv('BUCKET')

    client= storage.Client()
    bucket = client.get_bucket(BUCKET)
    blobs = bucket.list_blobs(prefix="output/tmp/", delimiter="/")

    c=0
    train_f1_scores = []
    test_f1_scores = []
    train_accuracy_scores = []
    test_accuracy_scores = []

    for blob in blobs:
        blobtmp=blob.name[11:]
        print(blobtmp)
        blob.download_to_filename(blobtmp)
        r=np.load(blobtmp)
        
        train_f1_scores.append(r[0])
        test_f1_scores.append(r[1])
        train_accuracy_scores.append(r[2])
        test_accuracy_scores.append(r[3])
        c+=1

    print("- Trainer Count : " + str(c) + " -")

    avg_train_f1 = np.mean(train_f1_scores)
    avg_test_f1 = np.mean(test_f1_scores)
    avg_train_accuracy = np.mean(train_accuracy_scores)
    avg_test_accuracy = np.mean(test_accuracy_scores)

    print(f"Average train F1 score: {avg_train_f1:.4f}")  # 0.9717
    print(f"Average test F1 score: {avg_test_f1:.4f}")  # 0.3333
    print(f"Average train accuracy: {avg_train_accuracy:.4f}")  # 0.9683
    print(f"Average test accuracy: {avg_test_accuracy:.4f}")  # 0.3535

    output = [avg_train_f1, avg_test_f1, avg_train_accuracy, avg_test_accuracy]

    print('- Complete -')

    return output

if __name__ == '__main__':
    exec_train_aggregate()
