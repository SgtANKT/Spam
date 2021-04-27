from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from autocorrect import Speller
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # tf_idf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
stemmer = PorterStemmer()
spell = Speller()


def preprocessing(data):
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    data.rename(columns={'v1': 'labels', 'v2': 'messages'}, inplace=True)
    data['labels'] = data['labels'].apply(lambda x: 1 if x == 'spam' else 0)

    return data


def wordcloud_plot(data):
    spam_words = ' '.join(list(data[data['labels'] == 'spam']['messages']))
    spam_wc = WordCloud(width=500, height=500).generate(spam_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(spam_wc)
    plt.show()

    spam_words = ' '.join(list(data[data['labels'] == 'ham']['messages']))
    spam_wc = WordCloud(width=500, height=500).generate(spam_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(spam_wc)
    plt.show()


def clean_txt(data):
    _data = preprocessing(data)
    new_data = []

    for i in tqdm(range(_data.shape[0])):
        lines = _data.iloc[i, 1]
        # removing non alphabatic characters
        lines = re.sub('[^A-Za-z]', ' ', lines)
        # lowering the every word
        lines = lines.lower()

        # tokenization
        tokenized_lines = word_tokenize(lines)

        # removing stop words ,stemming and spell correction
        processed_lines = []
        for i in tokenized_lines:
            if i not in set(stopwords.words('english')):
                processed_lines.append(spell(stemmer.stem(i)))

        final_lines = ' '.join(processed_lines)
        new_data.append(final_lines)
    return new_data



# Y.value_counts()


def train_and_recommend(data):
    new_data = clean_txt(data)
    # Splitting the data into
    Y = data['labels']
    X_train, X_test, Y_train, Y_test = train_test_split(new_data, Y, test_size=0.25)
    model = GaussianNB()
    # Vectorize data
    matrix = CountVectorizer()

    X_train_vect = matrix.fit_transform(X_train).toarray()
    X_test_vect = matrix.transform(X_test).toarray()

    # Fitting the data
    model.fit(X_train_vect, Y_train)
    # Predicting the outcomes for the test data
    Y_pred = model.predict(X_test_vect)

    score_value = accuracy_score(Y_test, Y_pred) * 100
    print(score_value)
    print(confusion_matrix(Y_test, Y_pred))

    pred_spams = pd.DataFrame(
        {'Predicted_label': Y_pred,
         'Actual_label': Y_test,
         'Messages': X_test
         })
    return pred_spams


if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\ankit\Desktop\Spam-Classifier-using-naive-bayes-main\spam.csv", encoding='ISO-8859-1')
    predictions = train_and_recommend(data)
    print(predictions)