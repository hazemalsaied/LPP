import logging
import os
import sys

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def raw2Lemmatized(rawP, lemmatizedP):
    wordnet_lemmatizer = WordNetLemmatizer()
    res = ''
    with open(rawP, 'r') as corpus:
        for line in corpus:
            tokens = tokenize(line)
            lemmas = []
            posTags = nltk.pos_tag(tokens)
            idx = 0
            for token in tokens:
                posTag = getWordnetPos(posTags[idx][1])
                if posTag:
                    lemmas.append(wordnet_lemmatizer.lemmatize(token, posTag))
                else:
                    lemmas.append(wordnet_lemmatizer.lemmatize(token))
                idx += 1
            res += ' '.join(lemmas) + '\n'
    with open(lemmatizedP, 'w') as corpus:
        corpus.write(res)


# def raw2Lemmatized(sourceCorpusP, idx=''):
#     wordnet_lemmatizer = WordNetLemmatizer()
#     res = ''
#     with open(os.path.join(sourceCorpusP, 'raw'+ str(idx) +'.txt'), 'r') as corpus:
#         for line in corpus:
#             tokens = tokenize(line)
#             lemmas = []
#             posTags = nltk.pos_tag(tokens)
#             idx = 0
#             for token in tokens:
#                 posTag = getWordnetPos(posTags[idx][1])
#                 if posTag:
#                     lemmas.append(wordnet_lemmatizer.lemmatize(token, posTag))
#                 else:
#                     lemmas.append(wordnet_lemmatizer.lemmatize(token))
#                 idx += 1
#             res += ' '.join(lemmas) + '\n'
#     with open(os.path.join(sourceCorpusP, 'lemmatized'+ str(idx) +'.txt'), 'w') as corpus:
#         corpus.write(res)

def tokenize(str):
    tokens = str.split(' ')
    for token in tokens:
        if not token.strip():
            tokens.remove(token)

    return tokens


def rawToConllu(rawFilePath, conlluPath):
    # Notice : use the default tag set
    conlluTxt = ''
    wordnet_lemmatizer = WordNetLemmatizer()
    # example = 'John\'s big idea isn\'t all that bad.\nYes, he did it!'
    with open(rawFilePath, 'r') as rawCorpus:
        for line in rawCorpus:
            tokens = tokenize(line[:-1])
            posTags = nltk.pos_tag(tokens)
            lemmaList = []
            for tag in posTags:
                if getWordnetPos(tag[1]):
                    lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0], pos=getWordnetPos(tag[1])))
                else:
                    lemmaList.append(wordnet_lemmatizer.lemmatize(tag[0]))
            if len(tokens) != len(posTags) and len(lemmaList) != len(tokens):
                print tokens
                print posTags
                print lemmaList
                raise
            for j in range(len(tokens)):
                conlluTxt += '{0}\t{1}\t{2}\t_\t{3}\t'.format(j + 1, tokens[j],
                                                              lemmaList[j].lower(),
                                                              # posTags[j][1]) + '_\t' * 4 + '_\n'
                                                              '_') + '_\t' * 4 + '_\n'
            conlluTxt += '\n'
    # print conlluTxt
    with open(conlluPath, 'w') as conllFile:
        conllFile.write(conlluTxt)
    logging.warn('The file {0} has been created!'.format(os.path.basename(conlluPath)))


def getWordnetPos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ

    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def getLexiconOccurence(rawCorpusP, lexiconP, lexiconWithFreqP):
    lexicon, result = {}, ''
    # Read the lexicon
    with open(lexiconP, 'r') as lexiconF:
        for line in lexiconF:
            tokens = tokenize(line)
            lexicon[' '.join(tokens)] = 0
    with open(rawCorpusP, 'r') as raxCorpus:
        for line in raxCorpus:
            tokens = tokenize(line)
            for entry in lexicon:
                entryTokens = tokenize(entry)
                entryTokensOccurences = []
                for entryToken in entryTokens:
                    entryTokensOccurences.append(len([i for i, j in enumerate(tokens) if j == entryToken]))
                entryOccurence = min(entryTokensOccurences)
                if entryOccurence:
                    lexicon[' '.join(entryTokens)] += entryOccurence
                    # print line, ' '.join(entryTokens), lexicon[' '.join(entryTokens)]
        freq = 0
        for k in sorted(lexicon.keys()):
            result += '{0},{1}\n'.format(k, lexicon[k])
            freq += lexicon[k]
        logging.warn('# Total number of mwe in the corpus = {}'.format(freq))
    with open(lexiconWithFreqP, 'w') as corpus:
        corpus.write(result)
    logging.warn('The file {0} has been created!'.format(os.path.basename(lexiconWithFreqP)))


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    # for i in range(1, 60):
    # raw2Lemmatized('../Corpora/CBT/Divided', str(i))
    # print [i for i, j in enumerate(['foo', 'bar', 'bar']) if j == 'bar']

    # getCBT2Gram()
    # getCBTRawStats()
    # getCBT2Gram()
    # getCBTStats(cbtRawTextPath,cbtRawStatsPath)
    # getCBTStats(cbt2gramPath, cbRaw2gramStatsPath)
    #
    # getLPPLexicon()
    # getLPPLexicon()
    # cbtRawTextPath = ''
    # cbtConllPath = ''

    cbtRawTextPath = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/CBT/raw.txt'
    cbtConllPath = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/CBT/conll-without-pos.txt'

    rawToConllu(cbtRawTextPath, cbtConllPath)
