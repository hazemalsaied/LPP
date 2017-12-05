# import modules & set up logging
import logging
import operator
import os
import sys

import gensim
from sklearn.metrics.pairwise import cosine_similarity

ANCPath = '../Corpora/OANC/data'
CBTPath = '../Corpora/CBT/Divided'

lppLexiconPath = '../Corpora/LPP/mweLEX.txt'

cbtModelPath = '../Word2Vec/word2vec.model'
cbt2gramModelPath = '../Word2Vec/word2vec.2gram.model'


def trainWord2Vec(gram2=False, sg=0, window=5):
    sentences, idx = [], 0
    lexicon = loadLexicon(lppLexiconPath)
    for R, D, F in os.walk(CBTPath):
        for txtF in F:
            with open(os.path.join(R, txtF), 'r') as rawFile:
                for line in rawFile:
                    line = cleanLine(line)
                    if gram2:
                        line = projectLexiconOnLine(line, lexicon)
                    tokens = tokenize(line)
                    if tokens:
                        sentences.append(tokens)
    print 'CBT sent num: ', len(sentences)
    for R, D, F in os.walk(ANCPath):
        for txtF in F:
            if txtF.endswith('.txt'):
                with open(os.path.join(R, txtF), 'r') as rawFile:
                    for line in rawFile:
                        line = cleanLine(line)
                        if gram2:
                            line = projectLexiconOnLine(line, lexicon)
                        tokens = tokenize(line)
                        if tokens:
                            sentences.append(tokens)
    print 'ANCP + CBT sent num: ', len(sentences)
    model = gensim.models.Word2Vec(sentences, sg=sg, window=window)
    if gram2:
        model.save(cbt2gramModelPath)
    else:
        model.save(cbtModelPath)


def getCompositionalityScores():
    lexicon = loadLexicon(lppLexiconPath)
    tokenModel = gensim.models.Word2Vec.load(cbtModelPath)
    gram2Model = gensim.models.Word2Vec.load(cbt2gramModelPath)
    result = {}
    for key in lexicon:
        tokens = key.split(' ')
        normalizedKey = key.replace(' ', '_')
        mWEInVocab = normalizedKey in gram2Model.wv.vocab
        allTokenInVocab = True
        sumTokenVector = [0] * 100
        for t in tokens:
            if t not in tokenModel.wv.vocab:
                allTokenInVocab = False
                break
            else:
                for i in range(100):
                    sumTokenVector[i] += tokenModel.wv[t][i]

        if mWEInVocab and allTokenInVocab:
            multTokenVector = [1] * 100
            for t in tokens:
                for i in range(100):
                    multTokenVector[i] *= tokenModel.wv[t][i]
            score = cosine_similarity(gram2Model.wv[normalizedKey], sumTokenVector)[0][0]
            multScore = cosine_similarity(gram2Model.wv[normalizedKey], multTokenVector)[0][0]
            result[key] = [score, multScore]
    sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    res = ''
    for k in sortedResult:
        res += k[0] + ',' + str(round(k[1][0], 4)) + ',' + str(round(k[1][1], 4)) + '\n'
    with open('../Word2Vec/com.csv', 'w') as f:
        f.write(res)


def compareCompositionality():
    lexicon = loadLexicon(lppLexiconPath)
    tokenModel = gensim.models.Word2Vec.load(cbtModelPath)
    gram2Model = gensim.models.Word2Vec.load(cbt2gramModelPath)
    for key in lexicon:
        tokens = key.split(' ')
        normalizedKey = key.replace(' ', '_')
        if normalizedKey not in gram2Model.wv.vocab:
            print key, -1
        else:
            print key, gram2Model.wv.vocab[normalizedKey]
        for t in tokens:
            if t not in tokenModel.wv.vocab:
                print key, -1
            else:
                print t, tokenModel.wv.vocab[t]


def loadLexicon(path):
    lexicon = set()
    with open(path, 'r') as lexiconF:
        for line in lexiconF:
            lexicon.add(line[:-1])
    return lexicon


# def testWord2VecModel():
#     nnKeysNum = 0
#     model = gensim.models.Word2Vec.load(cbt2gramModelPath)
#     print len(model.wv.vocab)
#     with open(lppLexiconPath, 'r') as lexicon:
#         for line in lexicon:
#             if len(line.split(' ')) == 2:
#                 key = line[:-1].lower().replace(' ', '_')
#                 if key not in model.wv.vocab:
#                     print key
#                     nnKeysNum += 1
#     print nnKeysNum
#     print len(model.wv.vocab)


def testWord2VecModel():
    model = gensim.models.Word2Vec.load(cbtModelPath)
    print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    # [('queen', 0.50882536)]
    print model.wv.doesnt_match("breakfast cereal dinner lunch".split())  # ,'cereal')
    print model.similarity('woman', 'man')
    print model.similarity('king', 'prince')

    print len(model.wv.vocab)


def projectLexiconOnLine(line, lexicon):
    line = line.lower()
    for k in lexicon:
        if k.lower() in line.lower():
            line = line.replace(k, k.replace(' ', '_'))
    return line


def cleanLine(line):
    line = line.lower()
    if line.endswith('\n'):
        line = line[:-1]
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    marks = [',', '!', '.', ':', '+', '?', '(', ')', '\\', '"', '#', '@', '\'', '&']
    tokens = line.split(' ')
    realTokens = []
    for t in tokens:
        if t and t not in numbers and t not in marks:
            for mark in marks:
                if mark in t:
                    t = t.replace(mark, '')
            for num in numbers:
                if num in t:
                    t = t.replace(num, '')
            t.replace('-', ' ')
            if t:
                realTokens.append(t)
    if realTokens:
        return ' '.join(realTokens)
    return ''


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sg, window = 1, 10
# trainWord2Vec(gram2=False, sg=sg, window=window)
# testWord2VecModel()
# trainWord2Vec(gram2=True, sg=sg, window=window)
# testWord2VecModel()
# getCompositionalityScores()
