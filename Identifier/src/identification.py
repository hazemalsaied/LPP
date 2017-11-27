import logging
import sys

import oracle
from classification import SVMClf
from corpus import *
from evaluation import evaluate
from extraction import extract
from parser import parse


def identify():
    corpus = Corpus(multipleFile=False)
    lexicon = {}
    for sent in corpus.trainingSents:
        for mwe in sent.vMWEs:
            if len(mwe.tokens) >1 :
                lexicon[getTokenText(mwe.tokens)] = True
    res  =''
    for k in sorted(lexicon.keys()):
        res += k+ '\n'
    with open('/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/mweLEX.txt', 'w') as F:
        F.write(res)
    return
    oracle.parse(corpus)
    labels, data = extract(corpus)
    svm = SVMClf(labels, data)
    parse(corpus, svm.classifier, svm.verctorizer)
    evaluate(corpus)
    print "condidence calculation"
    calculateConfidence(corpus)


def calculateConfidence(corpus):
    res, allMwes = '', ''
    for sent in corpus.testingSents:
        sent.getConfidence()
        mweStr = ''
        for mwe in sent.identifiedVMWEs:
            if len(mwe) > 1:
                mweStr += getTokenLemmas(mwe.tokens) + ',' + str(mwe.confidence) + '\n'
        res += sent.text + '\n' + mweStr if mweStr else ''
        allMwes += mweStr

    confP = '/Users/halsaied/PycharmProjects/LePetitPrince/src/hazem scores/LogisticRegression/Sent-confidence.txt'
    with open(confP, 'w') as confF:
        confF.write(res)
    confP = '/Users/halsaied/PycharmProjects/LePetitPrince/src/hazem scores/LogisticRegression/MWE-confidence.txt'
    with open(confP, 'w') as confF:
        confF.write(allMwes)


reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.WARNING)
identify()
