from sklearn.datasets import fetch_20newsgroups
from google.cloud import storage
import pickle

X_raw, y = fetch_20newsgroups(subset='all', return_X_y=True, shuffle=True, random_state=42, remove=("headers", "footers", "quotes"))  # X is a list of strings and y is a ndarray of category id

f = open('X_raw', 'wb')
pickle.dump(X_raw, f)
f = open("./X_raw","rb")
X_raw = pickle.load(f)

f = open('y', 'wb')
pickle.dump(y, f)
f = open("./y","rb")
y = pickle.load(f)

client= storage.Client()
bucket = client.bucket(BUCKET)
blob = bucket.blob("input/X_raw")
blob.upload_from_filename("./X_raw")
blob = bucket.blob("input/y")
blob.upload_from_filename("./y")