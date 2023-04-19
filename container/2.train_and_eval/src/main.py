import numpy as np
import os
import pickle
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage

def exec_train():
    # Env
    print('Start')
    BUCKET=os.getenv('BUCKET')
    num_tasks_trainer=int(os.environ.get('NUM_TASKS_TRAINER'))

    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("featurized/X.npz")
    blob.download_to_filename("./X.npz")
    
    print(BUCKET)

    client= storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("input/y")
    blob.download_to_filename("./y")

    # f = open("./X","rb")
    # X = pickle.load(f)
    X = sparse.load_npz("X.npz")

    f = open("./y","rb")
    y = pickle.load(f)

    split = list(KFold(n_splits=num_tasks_trainer, shuffle=True, random_state=42).split(y))

    train_f1_scores = []
    test_f1_scores = []
    train_accuracy_scores = []
    test_accuracy_scores = []
    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    for i in range(num_tasks_trainer):
        train_index, test_index = split[i]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)
        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)

    output_list =[train_f1_scores,test_f1_scores,train_accuracy_scores,test_accuracy_scores]
    output_name_list =['train_f1_scores','test_f1_scores','train_accuracy_scores','test_accuracy_scores']
    client= storage.Client()
    bucket = client.bucket(BUCKET)
    for i,o in enumerate(output_list):
        f = open(output_name_list[i], 'wb')
        pickle.dump(output_name_list[i], f)
        blob = bucket.blob("output/"+output_name_list[i])
        blob.upload_from_filename("./"+output_name_list[i])

    print('complete')

if __name__ == '__main__':
    exec_train()