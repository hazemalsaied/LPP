import datetime
import logging
import sys

from nltk.stem import WordNetLemmatizer

import projection


def crossLexiconOnRaw(corpusPath, crossedCorpusPath, csvFilePath, fieldNum=1):
    time = datetime.datetime.now()
    # Read the raw corpus
    scoreDic = readScores(csvFilePath, fieldNum)
    lexicon = readLexicon()
    with open(corpusPath, 'r') as lemmatizedCorpus:
        result = ''
        for line in lemmatizedCorpus:
            lemmas = tokenize(line[:-1])
            lemmaText = line[:-1]
            labels = ['0'] * len(lemmas)
            mweIdx = 1
            for entry in lexicon:
                entryScore = 0
                if entry in scoreDic:
                    entryScore = scoreDic[entry]
                if set(tokenize(entry)).issubset(set(lemmas)):
                    entryTokens = tokenize(entry)
                    if entry in lemmaText:
                        idxs = projection.getContinousIdxs(entryTokens, lemmas)
                        if idxs:
                            labels = annotate(labels, idxs, entryScore)
                            mweIdx += 1
                        continue
                    idxs = projection.entryInLine(entryTokens, lemmas)
                    hasLegalCont, inOneDirec = False, False
                    if idxs:
                        if len(entryTokens) < 3:
                            hasLegalCont = projection.hasLegalContinuty(idxs, windowsSize=1)
                        else:
                            hasLegalCont = projection.hasLegalContinuty(idxs, windowsSize=2)
                    if idxs and hasLegalCont:
                        inOneDirec = projection.inOneDirection(idxs, entry, lemmas)
                    if idxs and hasLegalCont and inOneDirec:
                        labels = annotate(labels, idxs, entryScore)
                        mweIdx += 1
            for i in range(len(lemmas)):
                result += '{0}\t{1}\n'.format(lemmas[i], labels[i])
            result += '\n'
    with open(crossedCorpusPath, 'w') as corpus:
        corpus.write(result)
    logging.warn(
        ' the file {0} has been made. It has taken {1}'.format(crossedCorpusPath, datetime.datetime.now() - time))


def annotate(labels, idxs, score):
    for idx in sorted(idxs):
        if idx == max(idxs):
            if labels[idx] == '0':
                labels[idx] = str(score)
            else:
                labels[idx] += ';' + str(score)
    return labels


def readLexicon():
    wordnet_lemmatizer = WordNetLemmatizer()
    lexicon = {}
    with open('../Corpora/LPP/mweLEX.txt', 'r') as scoresF:
        for line in scoresF:
            if line:
                entry = line[:-1].lower().strip()
                tokens = entry.split(' ')
                lemmas = []
                for token in tokens:
                    lemmas.append(wordnet_lemmatizer.lemmatize(token))
                lexicon[' '.join(lemmas)] = True
    return lexicon


def readScores(csvFilePath, fieldNum=1):
    lexicon = {}
    with open(csvFilePath, 'r') as scoresF:
        for line in scoresF:
            if line:
                parts = line.split(',')
                lexicon[parts[0].lower().strip()] = parts[fieldNum]
    return lexicon


def tokenize(line):
    tokens = line.lower().split(' ')
    realTokens = []
    for token in tokens:
        if token:
            realTokens.append(token)

    if realTokens:
        return realTokens
    return None


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    rawCopusPath = '../Corpora/LPP/lemmatized.txt'
    lppLexiconPath = '../Corpora/LPP/mweLEX.txt'
    resultFile = '../AnalysisFormat/Compositionality.txt'
    # csvFilePath = '../AssociationMeasures/CBT-scores/candidates-features.csv'
    csvFilePath = '../Word2Vec/CBOW/com.csv'

    crossLexiconOnRaw(rawCopusPath, resultFile, csvFilePath, fieldNum=1)
