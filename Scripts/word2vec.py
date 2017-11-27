# import modules & set up logging
import logging
import sys

import gensim
import nltk

cbtRawTextPath = '../Corpora/CBT-raw/data/cbt.txt'
cbt2gramPath = '../Corpora/CBT-raw/data/cbt-2gram.txt'
cbtModelPath = '../Corpora/word2vec.model'
cbt2gramModelPath = '../Corpora/word2vec.2gram.model'
lppLexiconPath = '../Corpora/LPPLexicon/lexicon.txt'


def getWord2vec(corpusPath, modelPath):
    with open(corpusPath, 'r') as cbtRaw:
        sentences = []
        for line in cbtRaw:
            tokens = nltk.word_tokenize(line.lower())
            sentences.append(tokens)

    model = gensim.models.Word2Vec(sentences)

    #
    model.save(modelPath)


def testCBT2GramModel():
    nnKeysNum = 0
    model = gensim.models.Word2Vec.load(cbt2gramModelPath)
    print len(model.wv.vocab)
    with open(lppLexiconPath, 'r') as lexicon:
        for line in lexicon:
            if len(line.split(' ')) == 2:
                key = line[:-1].lower().replace(' ', '_')
                if key not in model.wv.vocab:
                    print key
                    nnKeysNum += 1
    print nnKeysNum
    print len(model.wv.vocab)


def testCBTModel():
    model = gensim.models.Word2Vec.load(cbtModelPath)
    print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    # [('queen', 0.50882536)]
    print model.wv.doesnt_match("breakfast cereal dinner lunch".split())  # ,'cereal')
    print model.similarity('woman', 'man')
    print len(model.wv.vocab)


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# getWord2vec(cbtRawTextPath, cbtModelPath)
# getWord2vec(cbt2gramPath, cbt2gramModelPath)
testCBTModel()
testCBT2GramModel()
