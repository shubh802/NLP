import nltk
# nltk.download()
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """If you observe some people then you will notice that human life is a series of tension and problems. Also, they have a variety of concerns relating to their life. Sport is something that makes us free from these troubles, concerns, and tensions.
Moreover, they are an essential part of life who believe in life are able to face the problems. They help in the proper operation of various organs of the body. Furthermore, they refresh our mind and the body feel re-energized.
They also make muscle strength and keep them in good shape. In schools and colleges, they consider sports as an important part of education. Also, they organize sports competitions of different kinds.
In schools, they organize annual sports events. And on a daily basis, they have a specific period for sports and games. In this period teachers teach them the ways to play different sports and games.
These sports and games teach students new things and they have a bond with them. In addition, sports help them develop self-confidence and courage. Also, they become active and swift. And success fills them with motivation and eagerness.
We all knew the importance of games in the world. Consequently, now the Olympics (one of the biggest sports events) held in different countries. They held every fourth year. Moreover, the Asian Games is the biggest sports event on the Asian continent. Over, the year the interest of people in sports have increased many folds."""


sentences = nltk.sent_tokenize(paragraph)
# words = nltk.word_tokenize(paragraph)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming 
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
print(sentences)

# Lemmatizing
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
print(sentences)