from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv('C:\dataset\stocksentiment\Data.csv', encoding="ISO-8859-1")
df.head()

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

#Remove punctuation
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

#Renaming the column names
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head(5)

#Lower case
for index in new_Index:
    data[index] = data[index].str.lower()
data.head(2)

# We need all the reviews for one particular as one paragraph
# Below one is for one row
''.join(str(x) for x in data.iloc[1, 0:25])

headlines = []
for row in range(0, len(data.index)):
    headlines.append(''.join(str(x) for x in data.iloc[row, 0:25]))
headlines[2]


#BOW
countvector = CountVectorizer(ngram_range=(2, 2))
traindataset = countvector.fit_transform(headlines)

#RandomForest Classifier
randomclassifier = RandomForestClassifier(
    n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])

# Predict for the test dataset
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(''.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
print(predictions)

# Check the accuracy

matrix = confusion_matrix(test['Label'], predictions)
score = accuracy_score(test['Label'], predictions)
report = classification_report(test['Label'], predictions)
print("Matrix: ", matrix)
print("Score: ", score)
print(report)
