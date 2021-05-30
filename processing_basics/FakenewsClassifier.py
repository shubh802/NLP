# Dataset- https://www.kaggle.com/c/fake-news/data?select=train.csv

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import confusion_matrix
import pandas as pd
import re
import numpy as np
import itertools


df = pd.read_csv('C:\\dataset\\fakenews\\train.csv')
df.head()

# Get the Independent features and drop label
x = df.drop('label', axis=1)
x.head()

#Get the dependent feature
y = df['label']
y.head()
df.shape

# Text preprocessing
df = df.dropna()
df.shape
messages = df.copy()
messages.reset_index(inplace=True)
messages.head(10)
messages['title'][6]

# Removing stopwords
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# BOW
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
x = cv.fit_transform(corpus).toarray()

x.shape
y = messages['label']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0)

# To see the feature name
cv.get_feature_names()[:20]
cv.get_params()

count_df = pd.DataFrame(x_train, columns=cv.get_feature_names())
count_df.head()

# Custom Confusion matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#MultinomialNB Algo
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: ", score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Passive Aggresive Classifier

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter_no_change=50)

linear_clf.fit(x_train, y_train)
pred = linear_clf.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f"% score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])

# Multinomial Classifier with Hyperparameter
classifier = MultinomialNB(alpha=0.1)
previous_score = 0
for alpha in range(0,1,0.1):
    sub_classifier = MultinomialNB(alpha=alpha)
    sub_classifier.fit(x_train, y_train)
    pred = sub_classifier.predict(y_test)
    score = metrics.accuracy_score(y_test, pred)
    if score > previous_score:
        classifier = sub_classifier
    print("Alpha: {}, Score: {}".format(alpha, score))

# Get the feature names
feature_names = cv.get_feature_names()
classifier.coef_[0]

# Most Real value
sorted(zip(classifier.coef_[0], feature_names), reversed=True)[:20]

# Most Fake word
sorted(zip(classifier.coef_[0], feature_names))[:20]