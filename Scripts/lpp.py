import logging
import os
import sys

import nltk
from nltk.stem import WordNetLemmatizer

import utilities

lppDashedMWEPath = '../Corpora/LPP/mwe-dashed.txt'
lppRawPath = '../Corpora/LPP/raw.txt'
# @TODO
lppCrossedPath = '../Corpora/LPP/mwe.txt'
lppConlluPath = '../Corpora/LPP/conll.txt'

lppLexiconPath = '../Corpora/Lexicons/LPP/lexicon.txt'
lppLexiconLemmaPath = '../Corpora/Lexicons/LPP/lemma.txt'
lppLexiconStatsPath = '../Corpora/Lexicons/LPP/stats.txt'
lppLexiconWithMathieuFormatPath = '../Corpora/Lexicons/LPP/mathieu.txt'
lppFreqPath = '../Corpora/Lexicons/LPP/frequency.txt'

lexiconPatternXmlPath = '../mwetoolkit/LPP-patterns.xml'


class Chunck:
    def __init__(self, text):
        self.form = text
        self.words = nltk.word_tokenize(text.lower().strip().replace('_', ' '))
        self.lemmas = [''] * len(self.words)
        self.posTags = [''] * len(self.words)
        self.isExpression = True if len(self.words) > 1 else False


def getLPPRawCorpus():
    rawCorpus = ''
    with open(lppDashedMWEPath, 'r') as corpus:
        for line in corpus:
            rawCorpus += line.replace('_', ' ').strip() + '\n'
    with open(lppRawPath, 'w') as rawCorpusFile:
        rawCorpusFile.write(rawCorpus)
    logging.warn('The file {0} has been created!'.format(os.path.basename(lppRawPath)))


def getLPPLexicon():
    lexicon, lemmaLexicon = {}, {}
    wordnet_lemmatizer = WordNetLemmatizer()
    with open(lppDashedMWEPath, 'r') as corpus:
        for line in corpus:
            chuncks, words, lemmaList = [], [], []
            for token in nltk.word_tokenize(line[:-1]):
                chunck = Chunck(token)
                chuncks.append(chunck)
                words.extend(chunck.words)
            posTags = nltk.pos_tag(words)
            for tag in posTags:
                if utilities.getWordnetPos(tag[1]):
                    lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0], pos=utilities.getWordnetPos(tag[1])))
                else:
                    lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0]))
            idx = 0
            for chunck in chuncks:
                for i in range(len(chunck.words)):
                    chunck.lemmas[i] = lemmaList[idx]
                    chunck.posTags[i] = posTags[idx][1]
                    idx += 1
                if chunck.isExpression:
                    lemmaLexicon[' '.join(chunck.lemmas)] = True
                    lexicon[' '.join(chunck.words)] = True

    with open(lppLexiconPath, 'w') as lexiconFile:
        lexiconFile.write('\n'.join(sorted(lexicon.keys())))
    logging.warn('The file {0} has been created!'.format(os.path.basename(lppLexiconPath)))
    with open(lppLexiconLemmaPath, 'w') as lexiconFile:
        lexiconFile.write('\n'.join(sorted(lemmaLexicon.keys())))
    logging.warn('The file {0} has been created!'.format(os.path.basename(lppLexiconLemmaPath)))


def getLPPLexiconTStats():
    result = ''
    gram2, gram3, gram4, gram5, others = 0, 0, 0, 0, 0
    with open(lppLexiconPath, 'r') as lexicon:
        for line in lexicon:
            if len(line.split(' ')) == 2:
                gram2 += 1
            elif len(line.split(' ')) == 3:
                gram3 += 1
            elif len(line.split(' ')) == 4:
                gram4 += 1
            elif len(line.split(' ')) == 5:
                gram5 += 1
            else:
                others += 1
    result += '# of 2-gram MWEs = {0}\n'.format(gram2)
    result += '# of 3-gram MWEs = {0}\n'.format(gram3)
    result += '# of 4-gram MWEs = {0}\n'.format(gram4)
    result += '# of 5-gram MWEs = {0}\n'.format(gram5)
    result += '# of 6-gram MWEs = {0}\n'.format(others)
    with open(lppLexiconStatsPath, 'w') as lexiconStats:
        lexiconStats.write(result)
    logging.warn('The file {0} has been created!'.format(os.path.basename(lppLexiconStatsPath)))


def getLexiconWithMathieuFormat():
    lexiconMathieuFormat = ''
    with open(lppLexiconPath, 'r') as lexicon:
        for line in lexicon:
            lexiconMathieuFormat += line[:-1] + ',.X\n'
    with open(lppLexiconWithMathieuFormatPath, 'w') as mathieuLexicon:
        mathieuLexicon.write(lexiconMathieuFormat)
    logging.warn('The file {0} has been created!'.format(os.path.basename(lppLexiconWithMathieuFormatPath)))


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    # createPatternXml()
    # utilities.crossLexiconOnRaw(lppRawPath, lppLexiconPath, lppCrossedPath)
    # cbtRaw = '../Corpora/CBT/raw.txt'
    # cbtCrossedPath = '../Corpora/CBT/mwe.txt'
    # utilities.crossLexiconOnRaw(cbtRaw, lppLexiconPath, cbtCrossedPath)

    # utilities.getLexiconOccurence(lppRawPath,lppLexiconPath,lppFreqPath)
    # getLPPRawCorpus()
    # utilities.rawToConllu(lppRawPath, lppConlluPath)
    # getLPPLexicon()
    # getLPPLexiconTStats()
    # getLexiconWithMathieuFormat()
