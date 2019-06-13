from bs4 import BeautifulSoup
import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import requests
import re

# import pickle

data = open('/home/dipesh/PycharmProjects/NepaliPOStagging/pos_test_data.pos', 'r')
contents = data.read()
# print(type(contents))
soup = BeautifulSoup(contents, 'lxml')


def clean_soup(soup):
    for script in soup(["script"]):
        script.decompose()
    main_text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in main_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def tagged_corpus(corpus):
    return [list(''.join(sent).split() + ['।_SYM'])
            for sent in corpus.split('।_SYM')
            if len(sent) > 2]


def word_tag(corpus):
    corpus = [[','.join(text for text in word.split('_')) for word in sent] for sent in corpus]
    corpus = [[tuple(i for i in word.split(",")) for word in sent] for sent in corpus]
    return corpus


def untag(tagged_sentences):
    return [w[0] for w in tagged_sentences]


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    # print(sentence)
    # print(sentence[index])
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        # 'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        # 'is_all_caps': sentence[index].upper() == sentence[index],
        # 'is_all_lower': sentence[index].lower() == sentence[index],
        # 'prefix-1': sentence[index][0],
        # 'prefix-2': sentence[index][:2],
        # 'prefix-3': sentence[index][:3],
        # 'suffix-1': sentence[index][-1],
        # 'suffix-2': sentence[index][-2:],
        # 'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        # 'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


pprint.pprint(features(['१९०', 'यौं', 'भानुजयन्ती', 'प्रति', 'समर्पित', 'कविता', 'हरु', 'डा', '।'], 2))
nep_string = clean_soup(soup)
nep_rawcorpus = tagged_corpus(nep_string)
wordTag = word_tag(nep_rawcorpus)


def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for tagged in tagged_sentences:
        # print(tagged)
        # print("\n***********************************************************************\n")
        # print("\n***********************************************************************\n")

        for index in range(len(tagged)):
            # print(features(untag(tagged),index))
            # print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            X.append(features(untag(tagged), index))

            y.append(tagged[index][1])

            # print "index:"+str(index)+"original word:"+str(tagged)+"Word:"+str(untag(tagged))+"  Y:"+y[index]
    return X, y


cutoff = int(.75 * len(wordTag))
training_sentences = wordTag[:cutoff]
test_sentences = wordTag[cutoff:]

print(len(training_sentences)) # 2935(
print(len(test_sentences))

X, y = transform_to_dataset(training_sentences)
clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:1000],
        y[:1000])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print('Training completed')





X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test))


def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)


print(set(pos_tag(["पानी","सँग", "सँझौता","गर्न","सकेनछ",'।'])))
