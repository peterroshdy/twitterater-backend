import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.utils import shuffle
import pickle

data = pd.read_csv("./twitter_data.csv", error_bad_lines=False, delimiter=',', encoding = "ISO-8859-1")
#testing = pd.read_csv("./dataset_test.csv", error_bad_lines=False, delimiter=',')


train, test = train_test_split(data, test_size=0.33, shuffle=True)

train_x = list(train['text'])
train_y = list(train['score'])

test_x = list(test['text'])
test_y = list(test['score'])


# vectorizing the text to numbers so the machine can read it and get a pattern out of it
vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

print(clf_svm.score(test_x_vectors, test_y))

# Save the model as a pickle in a file 
with open('./models/classifier.pkl', 'wb') as f:
    pickle.dump(clf_svm, f)

with open('./models/victorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

