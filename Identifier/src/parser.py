from extraction import getFeatures
from transitions import *


def parse(corpus, clf, vectorizer):
    initializeSent(corpus)

    for sent in corpus.testingSents:
        sent.initialTransition = Transition(None, isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            newTransition = nextTrans(transition, sent, clf, vectorizer)
            newTransition.apply(transition, sent, parse=True, confidence=newTransition.confidence)
            transition = newTransition


def nextTrans(transition, sent, clf, vectorizer):
    legalTansDic = transition.getLegalTransDic()
    if len(legalTansDic) == 1:
        return initialize(legalTansDic.keys()[0], sent, confidence=1)

    featDic = getFeatures(transition, sent)
    if not isinstance(featDic, list):
        featDic = [featDic]

    probabilities = clf.predict_proba(vectorizer.transform(featDic))[0]
    confidence = max(probabilities)
    idxx = probabilities.tolist().index(confidence)
    print clf.classes_[idxx], '  ', confidence

    # print clf.predict_proba(vectorizer.transform(featDic))[0].index(
    #     max(clf.predict_proba(vectorizer.transform(featDic))[0]))
    # confidence =  max(clf.predict_proba(vectorizer.transform(featDic))[0])

    # confidence =  max(clf.decision_function(vectorizer.transform(featDic))[0])
    # print clf.decision_function(vectorizer.transform(featDic))[0]

    transTypeValue = clf.predict(vectorizer.transform(featDic))[0]
    print transTypeValue
    transType = getType(transTypeValue)
    if transType in legalTansDic:
        trans = legalTansDic[transType]
        trans.confidence = confidence
        return trans
    if len(legalTansDic):
        return initialize(legalTansDic.keys()[0], sent, confidence=1)
    raise


def initializeSent(corpus):
    for sent in corpus.testingSents:
        sent.identifiedVMWEs = []
        sent.initialTransition = None

        # print [1,2,3].index(2)
