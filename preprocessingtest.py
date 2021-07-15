import numpy as np,pandas as pd,string
from nltk.corpus import wordnet,stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import pos_tag
import joblib
import nltk


lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop = stopwords.words('english')
punc = list(string.punctuation)
stop = stop + punc

count_vec = joblib.load('count_vec.pkl')
rfc = joblib.load('rf.pkl')
nb = joblib.load('nb.pkl')
lg = joblib.load('lg.pkl')

class PredictFeatures:

    def __init__(self, doc):
        self.doc = doc

    def get_simple_pos_tag(self,tag):
        if tag[0]=='J':
            return wordnet.ADJ
        elif tag[0]=='V':
            return wordnet.VERB
        elif tag[0]=='N':
            return wordnet.NOUN
        elif tag[0]=='R':
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def clean_reviews(self, words):
        output_words = []
        for w in words:
            if w.lower() not in stop:
                ps = pos_tag([w])
                clean_word = lemmatizer.lemmatize(w,pos=self.get_simple_pos_tag(ps[0][1]))
                output_words.append(clean_word.lower())
        return output_words


    

    

    def predict(self):
        docum_test =list(word_tokenize(self.doc))

        docum_test = self.clean_reviews(docum_test)
        test_s = ''
        for i in docum_test:
            test_s += i+' '

        
        x_test_features = count_vec.transform([test_s])

        predict_ = []
        predict_.append(rfc.predict(x_test_features))
        predict_.append(nb.predict(x_test_features))
        predict_.append(lg.predict(x_test_features))

        count_n = 0
        count_p = 0
        for i in predict_:
            if i == 'negative':
                 count_n+=1
            else:
                count_p += 1
        print(count_n, count_p)
        if count_n > count_p:
            return 0
        else:
            return 1



