################################################################

# MMAI 5400 Assignment 2 -- Sentiment Classification
# Name: Qu Wen (Annita)
# Student ID: 250906656

# The loading of GridSearchCV might be taking more than 60 sec

################################################################

# import libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

################################################################
# 1. Loads reviews.csv

data = pd.read_csv('reviews.csv', delimiter='\t')

# Sentiment: 1, 2 = negative (0) | 3 = neutral(1) | 4, 5 = positive(2)
sentiment = data.drop(columns=['Name', 'DatePublished'])
sentiment.loc[sentiment.RatingValue >= 4, 'RatingValue'] = "Positive"
sentiment.loc[sentiment.RatingValue == 3, 'RatingValue'] = "Neutral"
sentiment.RatingValue = sentiment.RatingValue.replace([1, 2], 0)
sentiment.RatingValue = sentiment.RatingValue.replace(['Positive'], 2)
sentiment.RatingValue = sentiment.RatingValue.replace(['Neutral'], 1)
sentiment = sentiment.rename(columns={'RatingValue': 'Sentiment'})

# Drop positive ratings in order to balance the data
# to have equal numbers of negative, neutral and positive ratings
positive, neutral, negative = sentiment.Sentiment.value_counts()
min_count = min(positive, neutral, negative)    # Minimum count = 158

# print the shape of each class
positive_class = sentiment[sentiment['Sentiment'] == 2]
neutral_class = sentiment[sentiment['Sentiment'] == 1]
negative_class = sentiment[sentiment['Sentiment'] == 0]

# under sample each category
positive_undered = positive_class.sample(min_count)
neutral_undered = neutral_class.sample(min_count)
negative_undered = negative_class.sample(min_count)

# Concat the under sampled datafrmae (positive, neutral, and negative) together
sentiment_undered = pd.concat(
    [positive_undered,
     neutral_undered,
     negative_undered], axis=0)
sent_undered = sentiment_undered.reset_index(drop=True)

################################################################
# preprocesses the data


def clean_punc_noise(x):                # Data cleanning

    x = str(x)                          # set text to strings
    x = x.replace('\r', ' ')            # remove line breaks at the beginning
    x = x.replace('\n', ' ')            # remove line breaks at the end
    x = x.replace('', '')
    for eachpunc in punctuation:
        x = x.replace(eachpunc, ' ')    # remove punctuations
    x = re.sub(r'\s+', ' ', x)          # remove white spaces
    return x.lower()                    # set string to lower case

regexp = RegexpTokenizer('\w+')
sent_undered.Review_token = sent_undered.Review.map(clean_punc_noise)
sent_undered.Review_token = sent_undered.Review_token.apply(regexp.tokenize)

################################################################
# splits it and saves the files as train.csv and valid.csv

X = sent_undered.Review_token
y = sent_undered.Sentiment
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=42,
                                                    shuffle=True)
train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
valid = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

# save training and validation sets as train.csv and valid.csv
train.to_csv('train.csv')
valid.to_csv('valid.csv')

################################################################
# 2. Loads saved csv files

# Read the two csv files
training_set = pd.read_csv('train.csv')
validation_set = pd.read_csv('valid.csv')
# training_set.head(), validation_set.head()

X_train = training_set.Review
y_train = training_set.Sentiment
X_test = validation_set.Review
y_test = validation_set.Sentiment

################################################################
# 3. Build and train the model

SGD_pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier()),
    ]
)

parameters = {
    "vect__max_df": (0.5, 0.75, 1.0),
    'vect__max_features': (None, 2000, 5000),

    "vect__ngram_range": ((1, 2), (1, 3), (1, 4)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),

    "clf__max_iter": (80,),
    "clf__alpha": (0.00001, 0.000001),
    "clf__penalty": ("l2", "elasticnet"),
    # 'clf__max_iter': (10, 50, 80),
    }

grid_search = GridSearchCV(SGD_pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("SGD_pipeline:", [name for name, _ in SGD_pipeline.steps])
print("parameters:")
pprint(parameters)

print("Grid Search starts ......")
grid_search.fit(X_train, y_train)
print("Grid Search Finished")

print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

################################################################
# 4. Print perfromance metrics

# Prints the performance metrics on the validation set.
print("Best score: %0.3f" % grid_search.best_score_)
y_pred = grid_search.best_estimator_.predict(X_test)
print('Best accuracy: %0.3f' % metrics.accuracy_score(y_test, y_pred))
print('Best F1_score: %0.3f' % metrics.f1_score(y_test,
                                                y_pred,
                                                average='micro'))

target_names = ['negative', 'neutral', 'positive']
labels_names = [0, 1, 2]

print(classification_report(y_test,
                            y_pred,
                            labels=labels_names,
                            target_names=target_names))

cm = confusion_matrix(y_test, y_pred, labels=grid_search.classes_)
cmtx = pd.DataFrame(
    metrics.confusion_matrix(valid['Sentiment'], y_pred),
    index=['negative', 'neutral', 'positive'],
    columns=['negative', 'neutral', 'positive']
    )
print("Best Estimator's Confusion Matrix: \n", cmtx)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=grid_search.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='g')
plt.show()
