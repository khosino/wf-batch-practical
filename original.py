import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack

num_tasks_featurizer = 10
num_tasks_trainer = 5

# 1. Generate a classification problem
X_raw, y = fetch_20newsgroups(subset='all', return_X_y=True, shuffle=True, random_state=42, remove=("headers", "footers", "quotes"))  # X is a list of strings and y is a ndarray of category id

# 2. Divide the problem into folds
split = list(KFold(n_splits=num_tasks_trainer, shuffle=True, random_state=42).split(y))

# 3. Featurize the data using TF-IDF
batch_size = len(X_raw) // num_tasks_featurizer + 1
vectorizer = TfidfVectorizer(max_features=300)
vectorizer.fit(X_raw)
X = []
for i in range(10):
    X.append(vectorizer.transform(X_raw[i * batch_size:(i + 1) * batch_size]))
X = vstack(X)  # Convert X from a list of csr_matrix to just a csr_matrix

# 4. Train and evaluate the model per fold
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

# 5. Aggregate the F1 scores and accuracy for training and test data
avg_train_f1 = np.mean(train_f1_scores)
avg_test_f1 = np.mean(test_f1_scores)
avg_train_accuracy = np.mean(train_accuracy_scores)
avg_test_accuracy = np.mean(test_accuracy_scores)

print(f"Average train F1 score: {avg_train_f1:.4f}")  # 0.9717
print(f"Average test F1 score: {avg_test_f1:.4f}")  # 0.3333
print(f"Average train accuracy: {avg_train_accuracy:.4f}")  # 0.9683
print(f"Average test accuracy: {avg_test_accuracy:.4f}")  # 0.3535
