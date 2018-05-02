from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print (len(news.data))
print (len(news.target_names))
for text, num_label in zip(news.data[:10], news.target[:10]):
  print ('%s \t\t %s' % (news.target_names[num_label], text[:100].split('\n')[0]))

# split dataset to train and test datasets
from sklearn.cross_validation import train_test_split
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    classifier.fit(X_train, y_train)
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier
  

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
trial2 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', MultinomialNB()),
])
 
train(trial2, news.data, news.target)
