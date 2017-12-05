import logging
import os
import sys

import nltk

import utilities

cbtRawTextPath = '../Corpora/CBT/raw.txt'
cbtRawStatsPath = '../Corpora/CBT/stats.txt'
cbt2gramPath = '../Corpora/CBT/2gram.txt'
cbRaw2gramStatsPath = '../Corpora/CBT/2gram-stats.txt'
cbtConllPath = '../Corpora/CBT/conll.txt'


def getCBTStats(corpusPath, statsPath):
    result = ''
    wordNum, lineNum = 0, 0
    vocab = {}
    with open(corpusPath, 'r') as cbtRawFile:
        for line in cbtRawFile:
            words = nltk.word_tokenize(line)
            for word in words:
                if word.lower() not in vocab:
                    vocab[word] = True
            wordNum += len(words)
            lineNum += 1
    result += '# of word = {0}\n'.format(wordNum)
    result += '# of lines = {0}\n'.format(lineNum)
    result += 'Vocabulary size = {0}\n'.format(len(vocab))
    with open(statsPath, 'w') as lexiconStats:
        lexiconStats.write(result)
    logging.warn('The file {0} has been created!'.format(os.path.basename(statsPath)))


def getCBT2Gram():
    # @TODO
    with open(cbtRawTextPath, 'r') as cbtRawFile:
        result = ''
        for line in cbtRawFile:
            tokens = nltk.word_tokenize(line)
            if not tokens or len(tokens) <= 1:
                result += line
                continue
            for i in range(len(tokens) - 1):
                result += tokens[i] + '_' + tokens[i + 1] + ' '
            result += '\n'
    with open(cbt2gramPath, 'w') as cbt2gramFile:
        cbt2gramFile.write(result)
    logging.warn('The file {0} has been created!'.format(os.path.basename(cbt2gramPath)))


def raw2Conll(rawFolder, conllFolder):
    for rawF in os.listdir(rawFolder):
        print 'Workin with: ', rawF
        rawFP = os.path.join(rawFolder, rawF)
        idx = int(str(rawF).split('.')[0][3:])
        conllFP = os.path.join(conllFolder, 'conll{0}.txt'.format(idx))
        utilities.rawToConllu(rawFP, conllFP)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    # getCBTStats(cbtRawTextPath, cbtRawStatsPath)
    # utilities.rawToConllu(cbtRawTextPath, cbtConllPath)
    # getCBT2Gram()
    # getLemmatized('../Corpora/LPP')
    projectPath = '/Users/halsaied/PycharmProjects/LePetitPrince/'
    corpusPath = os.path.join(projectPath, 'Corpora/CBT')
    conllFolder = os.path.join(corpusPath, 'Divided-conll')
    rawFolder = os.path.join(corpusPath, 'Divided')
    raw2Conll(rawFolder, conllFolder)
