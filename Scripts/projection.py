import datetime
import logging
import os
import sys

from nltk.stem import WordNetLemmatizer

import utilities


def project(corpusP, lexiconPs, divideRaw=False):
    dividedP = os.path.join(corpusP, 'Divided')
    rawP = os.path.join(corpusP, 'raw.txt')
    if divideRaw:
        divideRawFile(rawP, dividedP)
    lexicon = {}
    for lexiconP in lexiconPs:
        lexicon.update(getLexicon(os.path.join('../Corpora/Lexicons/mwe-dictionaries', lexiconP)))
    print len(lexicon), 'lexicons size: '
    for subCorpus in os.listdir(dividedP):
        print 'Working with : ', subCorpus
        idx = int(str(subCorpus).split('.')[0][3:])
        subCorpusP = os.path.join(dividedP, subCorpus)
        crossedCorpusP = os.path.join(corpusP, 'Divided-MWE', 'mwe' + str(idx) + '.txt')
        lemmatizedP = os.path.join(corpusP, 'Divided-lemmas', 'lemma' + str(idx) + '.txt')
        subexiconP = os.path.join(corpusP, 'Divided-lex', 'lex' + str(idx) + '.txt')
        if not os.path.isfile(lemmatizedP):
            utilities.raw2Lemmatized(subCorpusP, lemmatizedP)
            print subCorpus, ' lemmatized!'
        if not os.path.isfile(subexiconP):
            print 'Lexicons are being filtered according to ', subCorpus
            subLexicon = filterDicWithVocabulary(lemmatizedP, lexicon)
            subLexicon = filterDicWithLineWordBag(lemmatizedP, subLexicon)
            with open(subexiconP, 'w') as subLExF:
                subLExF.write('\n'.join(subLexicon.keys()))
        else:
            subLexicon = {}
            with open(subexiconP, 'r') as subLExF:
                for line in subLExF:
                    subLexicon[line[:-1]] = True
            print 'Lexicon has been loaded!'
            print len(subLexicon), ' entries in lexicon'

        crossLexiconOnRaw(subCorpusP, subLexicon, crossedCorpusP)
        print 'projection is done', subCorpus


def lPPProject(lexiconPs):
    lexicon = {}
    for lexiconP in lexiconPs:
        lexicon.update(getLexicon(os.path.join('../Corpora/Lexicons/mwe-dictionaries', lexiconP)))
    print len(lexicon), 'lexicons size: '
    # print 'Working with : ', subCorpus
    lemmatizedP = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/lemmatized.txt'
    crossedCorpusP = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/mwe.txt'

    lexicon = filterDicWithVocabulary(lemmatizedP, lexicon)
    print 'Lexicon has been filtered!'
    print len(lexicon), ' entries in Lexicon!'
    crossLexiconOnRaw(lemmatizedP, lexicon, crossedCorpusP)
    print 'projection is done!'


def crossLexiconOnRaw(lemmmatizedCorpusP, lexicon, crossedCorpusP):
    time = datetime.datetime.now()
    # Read the raw corpus
    with open(lemmmatizedCorpusP, 'r') as lemmatizedCorpus:
        result = ''
        for line in lemmatizedCorpus:
            lemmas = tokenize(line[:-1])
            lemmaText = line[:-1]
            labels = ['_'] * len(lemmas)
            mweIdx = 1
            for entry in lexicon:
                if set(tokenize(entry)).issubset(set(lemmas)):
                    entryTokens = tokenize(entry)
                    if entry in lemmaText:
                        idxs = getContinousIdxs(entryTokens, lemmas)
                        if idxs:
                            labels = annotate(labels, idxs, mweIdx)
                            mweIdx += 1
                        continue
                    idxs = entryInLine(entryTokens, lemmas)
                    hasLegalCont, inOneDirec = False, False
                    if idxs:
                        if len(entryTokens) < 3:
                            hasLegalCont = hasLegalContinuty(idxs, windowsSize=1)
                        else:
                            hasLegalCont = hasLegalContinuty(idxs, windowsSize=2)
                    if idxs and hasLegalCont:
                        inOneDirec = inOneDirection(idxs, entry, lemmas)
                    if idxs and hasLegalCont and inOneDirec:
                        labels = annotate(labels, idxs, mweIdx)
                        mweIdx += 1
            for i in range(len(lemmas)):
                result += '{0}\t{1}\t_\t{2}\n'.format(i + 1, lemmas[i], labels[i])
            result += '\n'
    with open(crossedCorpusP, 'w') as corpus:
        corpus.write(result)
    logging.warn(
        ' the file {0} has been made. It has taken {1}'.format(crossedCorpusP, datetime.datetime.now() - time))


def getLexicon(lexiconP):
    print os.path.basename(lexiconP), 'is being proccesed!'
    lexicon = dict()
    wordnet_lemmatizer = WordNetLemmatizer()
    ooneWordExp = 0
    with open(lexiconP, 'r') as lexiconF:
        lineNum = 0
        for line in lexiconF:
            lineParts = line.split(',')
            words = tokenize(lineParts[0])
            if len(words) == 1:
                ooneWordExp += 1
                continue
            lemmas = []
            for word in words:
                lemma = wordnet_lemmatizer.lemmatize(word)
                lemmas.append(lemma)
            lexicon[' '.join(lemmas)] = ''
            lineNum += 1
    print ooneWordExp, ' one word expressions! '
    print lineNum, ' non lemmatizes entries'
    print len(lexicon), ' lexicon length '
    return lexicon


def filterDicWithLineWordBag(corpusP, lexicon):
    for k in lexicon.keys():
        lexicon[k] = False
    with open(corpusP, 'r') as corpus:
        for line in corpus:
            lineWordBag = set(tokenize(line))
            for entry in lexicon:
                if set(tokenize(entry)).issubset(lineWordBag):
                    lexicon[entry] = True
    mweDic = dict()
    for k in lexicon.keys():
        if lexicon[k]:
            mweDic[k] = True
    print 'Lexicon after second filter: ', len(mweDic)
    return mweDic


def filterDicWithVocabulary(corpusP, lexicon):
    for k in lexicon.keys():
        lexicon[k] = False
    vocabulary = getVocabulary(corpusP)
    print 'Sub corpus dictionary size: ', len(vocabulary)
    mweDic = dict()
    for key in lexicon.keys():
        lemmaSet = set(tokenize(key))
        if lemmaSet.issubset(vocabulary):
            mweDic[key] = True
    print 'Lexicon after first filter: ', len(mweDic)
    return mweDic


def getVocabulary(corpusP):
    vocabulary = set()
    with open(corpusP, 'r') as corpus:
        for line in corpus:
            lemmas = tokenize(line[:-1])
            for lemma in lemmas:
                if lemma not in vocabulary:
                    vocabulary.add(lemma)
    return vocabulary


def filterDicHeavily(corpusP, lexiconP, newLexP):
    lexicon = {}
    with open(lexiconP, 'r') as lexiconF:
        for line in lexiconF:
            lexicon[line[:-1]] = False

    with open(corpusP, 'r') as corpus:
        for line in corpus:
            for entry in lexicon.keys():
                if entry in line:
                    lexicon[entry] = True
    for k in lexicon.keys():
        if not lexicon[k]:
            del lexicon[k]
    res = ''
    with open(newLexP, 'w') as newLex:
        for k in sorted(lexicon.keys()):
            res += k + '\n'
        newLex.write(res)


def divideRawFile(p, newP):
    idx = 0
    fileSize = 5000
    res = ''
    fileNum = 1
    with open(p, 'r') as f:
        for line in f:
            if idx >= fileSize * fileNum:
                with open(os.path.join(newP, os.path.basename(p)[:-4] + str(fileNum) + '.txt'), 'w+') as newFile:
                    newFile.write(res)
                res = ''
                fileNum += 1
            res += line
            idx += 1

        with open(os.path.join(newP, os.path.basename(p)[:-4] + str(fileNum) + '.txt'), 'w+') as newFile:
            newFile.write(res)
    print 'The corpus was divided into ', fileNum, 'raw files'


def annotate(labels, idxs, mweIdx):
    for idx in sorted(idxs):
        if idx == min(idxs):
            strToAdd = str(mweIdx) + ':OTH'
        else:
            strToAdd = str(mweIdx)
        if labels[idx] == '_':
            labels[idx] = strToAdd
        else:
            labels[idx] += ';' + strToAdd
    return labels


def getContinousIdxs(entryTokens, lemmas):
    idxs = []
    for lemma in lemmas:
        if lemma == entryTokens[0]:
            idx = lemmas.index(lemma)
            idxs.append(idx)
            isContinous = True
            if idx <= len(lemmas) - len(entryTokens):
                for i in range(1, len(entryTokens)):
                    if entryTokens[i] != lemmas[i + idx]:
                        isContinous = False
                        break
                    idxs.append(i + idx)
            if isContinous:
                return idxs
            else:
                idxs = []
    return None


def inOneDirection(idxs, entry, lemmas):
    idxs = sorted(idxs)
    result = ''
    for idx in idxs:
        result += lemmas[idx] + ' '
    if result[:-1] != entry:
        return False
    return True


def entryInLine(entryTokens, tokens):
    idxs = []
    for entryToken in entryTokens:
        if entryToken not in tokens:
            return []
        idxs.append(tokens.index(entryToken))
    return idxs


def hasLegalContinuty(idxs, windowsSize=2):
    idxs = sorted(idxs, reverse=True)
    for i in range(len(idxs) - 1):
        if idxs[i] - idxs[i + 1] > windowsSize:
            return False
    return True


def tokenize(str):
    tokens = str.split(' ')
    for token in tokens:
        if not token.strip():
            tokens.remove(token)

    return tokens


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    lPPProject(['adam-mwe.txt', 'unitex-mwe.txt'])


    # project('../Corpora/CBT/', ['adam-mwe.txt', 'unitex-mwe.txt'])
    #
    # projectPath = '/Users/halsaied/PycharmProjects/LePetitPrince/'
    # corpusPath = os.path.join(projectPath, 'Corpora/CBT')
    # conllFolder = os.path.join(corpusPath, 'Divided-conll')
    # rawFolder = os.path.join(corpusPath, 'Divided')
    # cbt.raw2Conll(rawFolder, conllFolder)
