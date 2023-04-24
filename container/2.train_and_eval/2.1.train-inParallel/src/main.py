import numpy as np
import os
import pickle
from scipy import sparse
# from sklearn.model_selection import KFold
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# import cuml
from cuml.model_selection import KFold
from cuml.metrics import f1_score, accuracy_score
from cuml.ensemble import RandomForestClassifier

from google.cloud import storage
import time

def exec_train():
    # Env
    BUCKET=os.getenv('BUCKET')
    INPUT_BUCKET=os.getenv('INPUT_BUCKET')
    BATCH_TASK_INDEX=int(os.environ.get('BATCH_TASK_INDEX'))
    num_tasks_trainer=int(os.environ.get('NUM_TASKS_TRAINER'))

    print("--- BatchTaskIndex : " + str(BATCH_TASK_INDEX+1) + " Start ---")

    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("featurized/X.npz")
    blob.download_to_filename("./X.npz")

    client= storage.Client()
    bucket = client.bucket(INPUT_BUCKET)
    blob = bucket.blob("input/y")
    blob.download_to_filename("./y")

    X = sparse.load_npz("X.npz")
    f = open("./y","rb")
    y = pickle.load(f)

    split = list(KFold(n_splits=num_tasks_trainer, shuffle=True, random_state=42).split(y))

    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf = RandomForestClassifier(n_jobs=-1, random_state=42)

    train_index, test_index = split[BATCH_TASK_INDEX]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    result_list = [train_f1,test_f1,train_accuracy,test_accuracy]

    result_list_name = 'result_list_'+str(BATCH_TASK_INDEX)

    np.save(result_list_name,result_list)
    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("output/tmp/"+result_list_name+".npy")
    blob.upload_from_filename(result_list_name+".npy")

    print('Complete')

if __name__ == '__main__':
    start = time.time()
    exec_train()
    process_time = time.time() - start
    print("Time: "+str(process_time))