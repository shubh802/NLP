import nltk

paragraph = """If you observe some people then you will notice that human life is a series of tension and problems. Also, they have a variety of concerns relating to their life. Sport is something that makes us free from these troubles, concerns, and tensions.
Moreover, they are an essential part of life who believe in life are able to face the problems. They help in the proper operation of various organs of the body. Furthermore, they refresh our mind and the body feel re-energized.
They also make muscle strength and keep them in good shape. In schools and colleges, they consider sports as an important part of education. Also, they organize sports competitions of different kinds.
In schools, they organize annual sports events. And on a daily basis, they have a specific period for sports and games. In this period teachers teach them the ways to play different sports and games.
These sports and games teach students new things and they have a bond with them. In addition, sports help them develop self-confidence and courage. Also, they become active and swift. And success fills them with motivation and eagerness.
We all knew the importance of games in the world. Consequently, now the Olympics (one of the biggest sports events) held in different countries. They held every fourth year. Moreover, the Asian Games is the biggest sports event on the Asian continent. Over, the year the interest of people in sports have increased many folds."""

#Cleaning the text
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

# TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()
print(x.shape)