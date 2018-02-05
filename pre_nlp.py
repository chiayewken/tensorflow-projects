
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize as tokenize
import numpy as np
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
line_limit = 100000


# In[2]:


def create_lexicon(pos,neg):
    lexicon = []
    
    def add_words(file, lexicon):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines[:line_limit]:
                words = tokenize(line)
                lexicon += list(words)
                
    add_words(pos, lexicon)
    add_words(neg, lexicon)
            
    lexicon = [lemma.lemmatize(word) for word in lexicon]
    counted_lexicon = []
    count = Counter(lexicon)
    for word in count:
        if 1000>count[word]>50:
            counted_lexicon.append(word)
    lexicon = counted_lexicon
    print('lexicon length = {}'.format(len(lexicon)))
    return lexicon
            


# In[3]:


def create_features(sample,lexicon,classification):
    featureset = []
    with open(sample, 'r') as file:
        contents = file.readlines()
        for line in contents[:line_limit]:
            current_words = tokenize(line.lower())
            current_words = [lemma.lemmatize(word) for word in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    index = lexicon.index(word)
                    features[index] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset


# In[4]:


def featureset(pos,neg,test_fraction):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += create_features(pos,lexicon,[1,0])
    features += create_features(neg,lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)
    
    test_size = int(test_fraction*len(features))
    
    trainx = list(features[:,0][test_size:])
    trainy = list(features[:,1][test_size:])
    testx = list(features[:,0][:test_size])
    testy = list(features[:,1][:test_size])
    print('total sample size = {}'.format(len(features)))
    print('train = {}, test = {}' .format(len(trainx), len(testx)))
    return trainx,trainy,testx,testy


# In[5]:


#pos = 'C:/Users/chiay/Downloads/git/tensorflow/pos.txt'
#neg = 'C:/Users/chiay/Downloads/git/tensorflow/neg.txt'
#trainx,trainy,testx,testy = featureset(pos,neg,test_fraction=0.1)


# In[ ]:




