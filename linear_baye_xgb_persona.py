import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA, FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import xgboost as xgb

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


import pickle

data = pd.read_csv("data/mbti_1.csv")

[p.split('|||') for p in data.head(1).posts.values]

plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=data, x='type')

plt.savefig("plots/Fig1_type_distribution.png")


unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

lab_encoder = LabelEncoder().fit(unique_type_list)


data.posts[1].replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data.posts[1])

# Lemmatizer | Stemmatizer
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
cachedStopWords = stopwords.words("english")

# One post
OnePost = data.posts[1]

# List all urls
urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', OnePost)

# Remove urls
temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', OnePost)

# Keep only words
temp = re.sub("[^a-zA-Z]", " ", temp)

# Remove spaces > 1
temp = re.sub(' +', ' ', temp).lower()

# Remove stopwords and lematize
stemmer.stem(" ".join([w for w in temp.split(' ') if w not in cachedStopWords]))

print("\nBefore preprocessing:\n\n", OnePost[0:500])
print("\nAfter preprocessing:\n\n", temp[0:500])
print("\nList of urls:")
urls

#### Compute list of subject with Type | list of comments
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()


def pre_process_data(data, remove_stop_words=True):
       list_personality = []
       list_posts = []
       len_data = len(data)
       i = 0

       for row in data.iterrows():
              i += 1
              if i % 500 == 0:
                     print("%s | %s rows" % (i, len_data))

              ##### Remove and clean comments
              posts = row[1].posts
              temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
              temp = re.sub("[^a-zA-Z]", " ", temp)
              temp = re.sub(' +', ' ', temp).lower()
              if remove_stop_words:
                     temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
              else:
                     temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

              type_labelized = lab_encoder.transform([row[1].type])[0]
              list_personality.append(type_labelized)
              list_posts.append(temp)

       # del data
       list_posts = np.array(list_posts)
       list_personality = np.array(list_personality)
       return list_posts, list_personality


list_posts, list_personality = pre_process_data(data, remove_stop_words=True)


#Vectorize with count and tf-idf

cntizer = CountVectorizer(analyzer="word",
                          max_features=1500,
                          tokenizer=None,
                          preprocessor=None,
                          stop_words=None,
                          #                             ngram_range=(1,1),
                          max_df=0.5,
                          min_df=0.1)

tfizer = TfidfTransformer()

print("CountVectorizer")
X_cnt = cntizer.fit_transform(list_posts)
print("Tf-idf")
X_tfidf = tfizer.fit_transform(X_cnt).toarray()

list_posts[0]

#Top-50 words
reverse_dic = {}
for key in cntizer.vocabulary_:
    reverse_dic[cntizer.vocabulary_[key]] = key

top_50 = np.asarray(np.argsort(np.sum(X_cnt, axis=0))[0,-50:][0, ::-1]).flatten()
[reverse_dic[v] for v in top_50]


#Dimenstionality reduction
svd = TruncatedSVD(n_components=12, n_iter=7, random_state=42)
svd_vec = svd.fit_transform(X_tfidf)

print("TSNE")
X_tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=650).fit_transform(svd_vec)



#Plot TSNE with 3 components
col = list_personality
plt.figure(0)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=col, cmap=plt.get_cmap('tab20') , s=12)
plt.savefig("plots/Fig2a_tsne_component1.png")

plt.figure(1)
plt.scatter(X_tsne[:,0], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
plt.savefig("plots/Fig2b_tsne_component2.png")

plt.figure(2)
plt.scatter(X_tsne[:,1], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("plots/Fig2c_tsne_component3.png")


#Plot first Axes of decomposition
# PCA
pca_vec = PCA(n_components=12).fit_transform(X_tfidf)

# ICA
ica_vec = FastICA(n_components=12).fit_transform(X_tfidf)

# Plot
plt.figure(1)
plt.scatter(pca_vec[:,0], pca_vec[:,1], c=list_personality, cmap=plt.get_cmap('tab20'), s=12)
plt.savefig("plots/Fig3a_pca.png")

plt.figure(2)
plt.scatter(svd_vec[:,0], svd_vec[:,1], c=list_personality, cmap=plt.get_cmap('tab20'), s=12)
plt.savefig("plots/Fig3b_svd.png")

plt.figure(3)
plt.scatter(ica_vec[:,0], ica_vec[:,1], c=list_personality, cmap=plt.get_cmap('tab20'), s=12)
plt.savefig("plots/Fig3a_ica.png")


#Plotting TSNE for each pair

# Split mbti personality into 4 letters and binarize
titles = ["Extraversion (E) - Introversion (I)",
          "Sensation (S) - INtuition (N)",
          "Thinking (T) - Feeling (F)",
          "Judgement (J) - Perception (P)"
         ]
b_Pers = {'I':1, 'E':0, 'N':1,'S':0, 'T':1, 'F':0, 'J':1, 'P': 0}
b_Pers_list = [{1:'I', 0:'E'},
               {1:'N', 0:'S'},
               {1:'T', 0:'F'},
               {1:'J', 0:'P'}]

def translate_personality(personality):
    '''
    transform mbti to binary vector
    '''
    return [b_Pers[l] for l in personality]

def translate_back(personality):
    '''
    transform binary vector to mbti personality
    '''
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

list_personality_bin = np.array([translate_personality(p) for p in data.type])
print("Binarize MBTI list: \n%s" % list_personality_bin)

# Plot
def plot_tsne(X, i):
    a = plt.figure(i, figsize=(30,20))
    plt.title(titles[i])
    plt.subplot(3,1,1)
    plt.scatter(X[:,0], X[:,1], c=list_personality_bin[:,i], cmap=plt.get_cmap('Dark2'), s=25)
    plt.subplot(3,1,2)
    plt.scatter(X[:,0], X[:,2], c=list_personality_bin[:,i], cmap=plt.get_cmap('Dark2'), s=25)
    plt.subplot(3,1,3)
    plt.scatter(X[:,1], X[:,2], c=list_personality_bin[:,i], cmap=plt.get_cmap('Dark2'), s=25)


#Extraversion-Introversion
plot_tsne(X_tsne, 0)
plt.savefig("plots/Fig4a_tsne_E-I.png")


#Sensation-Intuition
plot_tsne(X_tsne, 1)
plt.savefig("plots/Fig4b_tsne_S-N.png")

#Thinking-Feeling
plot_tsne(X_tsne, 2)
plt.savefig("plots/Fig4c_tsne_T-F.png")

#Judgement-Perception
plot_tsne(X_tsne, 3)
plt.savefig("plots/Fig4d_tsne_J-P.png")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("plots/"+title+".png")



# Vectorizer

cntizer = CountVectorizer(analyzer="word",
                             max_features=1000,
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.5,
                             min_df=0.1)

tfizer = TfidfTransformer()

# Classifiers
PassAgg = PassiveAggressiveClassifier(max_iter=50)

sgd = SGDClassifier(loss='hinge',
              penalty='l1',
              alpha=1e-2,
              random_state=42,
              max_iter=7,
              tol=None)

# SVM
lsvc = LinearSVC()

# Multinomial Naive Bayes
mlNB = MultinomialNB()

# Xgboost
# setup parameters for xgboost
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['learning_rate'] = 0.6
param['n_estimators'] = 300
param['subsample'] = 0.93
param['max_depth'] = 2
param['silent'] = 1
param['nthread'] = 8
xgb_class = xgb.XGBClassifier(**param)
param['num_class'] = len(unique_type_list)


#Stratified K-fold validation training

name = lambda x: str(x).split('(')[0]


def train_stratified(models, X, y, add_idf=False, nsplits=3, confusion=False):
    '''
    Take a sklearn model like, feature set X, target set y and number of splits to compute Stratified kfold validation.
    Args:
        X (array):       Numpy array of features.
        y (str):         Target - Personality list.
        add_idf (bool):  Wehther to use tf-idf on CountVectorizer.
        nsplits(int):    Number of splits for cross validation.
        confusion(bool): Wether to plot confusion matrix

    Returns:
        dict: Dictionnary of classifiers and their cv f1-score.
    '''
    fig_i = 0
    kf = StratifiedShuffleSplit(n_splits=nsplits)

    # Store fold score for each classifier in a dictionnary
    dico_score = {}
    dico_score['merged'] = 0
    for model in models:
        dico_score[name(model)] = 0

    # Stratified Split
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        X_train = cntizer.fit_transform(X_train)
        X_test = cntizer.transform(X_test)

        # tf-idf
        if add_idf == True:
            X_train_tfidf = tfizer.fit_transform(X_train)
            X_test_tfidf = tfizer.transform(X_test)

            X_train = np.column_stack((X_train_tfidf.todense(), X_train))
            X_test = np.column_stack((X_test_tfidf.todense(), X_test))

        probs = np.ones((len(y_test), 16))
        for model in models:
            # if xgboost use dmatrix
            if 'XGB' in name(model):
                xg_train = xgb.DMatrix(X_train, label=y_train)
                xg_test = xgb.DMatrix(X_test, label=y_test)
                watchlist = [(xg_train, 'train'), (xg_test, 'test')]
                num_round = 30
                bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=6)
                preds = bst.predict(xg_test)
                probs = np.multiply(probs, preds)
                preds = np.array([np.argmax(prob) for prob in preds])
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = np.multiply(probs, model.predict_proba(X_test))
            # f1-score
            score = f1_score(y_test, preds, average='weighted')
            dico_score[name(model)] += score
            print('%s : %s' % (str(model).split('(')[0], score))

            if confusion == True:
                # Compute confusion matrix
                cnf_matrix = confusion_matrix(y_test, preds)
                np.set_printoptions(precision=2)
                # Plot confusion matrix
                plt.figure(fig_i)
                fig_i += 1
                modelname = name(model)
                if(modelname == "MultinomialNB"):
                    modelname = "LinearSVM"
                plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(16)), normalize=True,
                                      title=('Confusion matrix %s' % modelname ))

        # product of class probabilites of each classifier
        merged_preds = [np.argmax(prob) for prob in probs]
        score = f1_score(y_test, merged_preds, average='weighted')
        print('Merged score: %s' % score)
        dico_score['merged'] += score

    return {k: v / nsplits for k, v in dico_score.items()}


#Compare multinomial naive bayes, xgb and their product predictions

results = train_stratified([mlNB, xgb_class], list_posts, list_personality, add_idf=False, nsplits=5, confusion=True)

results



# Try multi-output classification
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    '''
    transform mbti to binary vector
    '''
    return [b_Pers[l] for l in personality]

def translate_back(personality):
    '''
    transform binary vector to mbti personality
    '''
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

list_personality_bin = np.array([translate_personality(p) for p in data.type])
print("Binarize MBTI list: \n%s" % list_personality_bin)


# Feed classifier to MultiOutputCLassifier

clf = AdaBoostClassifier()
multi_target_classifier = MultiOutputClassifier(clf, n_jobs=-1)
multi_target_classifier.fit(X_tfidf, list_personality_bin)
preds = multi_target_classifier.predict(X_tfidf)

preds_t = [translate_back(p) for p in preds]
vec1 = data.type ==  preds_t
for i in range(4):
    print("f1 score for %s:\n%s" % (titles[i],
                                    f1_score(np.array(list_personality_bin)[:,i], preds[:,i])))

# Stratified cross val for multi-output
X = list_posts
y = np.array(list_personality_bin)

clf = AdaBoostClassifier()

kf = StratifiedShuffleSplit(n_splits=4)

list_score = []
list_score_per_class = []

for train, test in kf.split(X, y):
    X_train, X_test, y_train, y_test = \
        X[train], X[test], y[train], y[test]

    X_train = cntizer.fit_transform(X_train)
    X_test = cntizer.transform(X_test)

    X_train = tfizer.fit_transform(X_train).toarray()
    X_test = tfizer.transform(X_test).toarray()

    multi_target_classifier = MultiOutputClassifier(clf, n_jobs=-1)
    multi_target_classifier.fit(X_train, y_train)
    preds = multi_target_classifier.predict(X_test)

    rev_preds = np.array([translate_back(p) for p in preds])
    rev_test = np.array([translate_back(p) for p in y_test])
    score = f1_score(rev_test, rev_preds, average='weighted')
    list_score.append(score)
    print('\nTotal score: %s' % f1_score(rev_test, rev_preds, average='weighted'))

    list_temp = []
    for i in range(4):
        score_per_class = f1_score(y_test[:, i], preds[:, i])
        list_temp.append(score_per_class)
        print(score_per_class)
    list_score_per_class.append(list_temp)

list_score_per_class = np.array(list_score_per_class)
print('Mean score per classes: %s' % list_score_per_class.mean(axis=0))
