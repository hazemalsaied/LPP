import sys
import xml.etree.ElementTree as ET

from nltk.stem import WordNetLemmatizer

cbtScoresXML = '../mwetoolkit/CBT-scores/candidates-features.xml'
cbtScoresCSV = '../mwetoolkit/CBT-scores/candidates-features.csv'

lppScoresXML = '../mwetoolkit/LPP-scores/candidates-features.xml'
lppScoresCSV = '../mwetoolkit/LPP-scores/candidates-features.csv'

rawCandidates = '../mwetoolkit/CBT-scores/candidates.txt'

# lppLexiconPath = '../Corpora/Lexicons/LPP/lexicon.txt'
lexiconPatternXmlPath = '../mwetoolkit/LPP-patterns.xml'


def createPatternXml(lppLexiconPath, lexiconPatternXmlPath):
    wordnet_lemmatizer = WordNetLemmatizer()
    patternFileTxt = '<?xml version="1.0" encoding="UTF-8"?>'
    patternFileTxt += '<!DOCTYPE dict SYSTEM "dtd/mwetoolkit-patterns.dtd">'
    patternFileTxt += '<patterns>'
    with open(lppLexiconPath, 'r') as lexiconFile:
        for line in lexiconFile:
            line = line[:-1]
            patternFileTxt += '<pat>'
            for w in line.split(' '):
                if w.strip():
                    lemma = wordnet_lemmatizer.lemmatize(w)
                    patternFileTxt += '<w surface="{0}"></w>'.format(lemma)
            patternFileTxt += '</pat>'
    patternFileTxt += '</patterns>'
    with open(lexiconPatternXmlPath, 'w') as patternsFile:
        patternsFile.write(patternFileTxt)


def getScoresFromXML(readingPath, writingPath, prefix='cbt'):
    tree = ET.parse(readingPath)
    root = tree.getroot()
    result = {}
    for cand in root:
        if cand.tag == 'cand':
            mweLemmaStr, mweStr, posTags, frequency, features = '','','', 0, dict()
            for c in cand:
                if c.tag == 'ngram':
                    for w in c:
                        if w.tag == 'w':
                            # mweLemmaStr += w.attrib['lemma'] + ' '
                            if 'surface' in w.attrib:
                                mweStr += w.attrib['surface'] + ' '
                            else:
                                mweStr += w.attrib['lemma'] + ' '
                            # posTags += w.attrib['pos'] + ' '
                    # mweLemmaStr = mweLemmaStr[:-1]
                    mweStr = mweStr[:-1]
                    # posTags = posTags[:-1]
                if c.tag == 'occurs':
                    for ch in c[0]:
                        if ch.tag == 'freq':
                            frequency = ch.attrib['value']
                if c.tag == 'features':
                    for ch in c:
                        name = ch.attrib['name']
                        value = float(ch.attrib['value'])
                        features[name] = value
            # print mweStr, frequency, features
            result['{0} , {1} , {2} , {3} , {4} , {5} , {6}\n'.format(mweStr, frequency,
                                                             features['mle_' + prefix] if (
                                                                                          'mle_' + prefix) in features else'',
                                                             features['dice_' + prefix] if (
                                                                                           'dice_' + prefix) in features else'',
                                                             features['ll_' + prefix] if (
                                                             'll_' + prefix in features) else'',
                                                             features['t_' + prefix] if (
                                                             't_' + prefix in features) else'',
                                                             features['pmi_' + prefix] if (
                                                             'pmi_' + prefix in features) else'')] = True

    with open(writingPath, 'w') as csvFile:
        csvFile.write(''.join(sorted(result.keys())))


def getCandidatesFromXML(readingPath, writingPath):
    tree = ET.parse(readingPath)
    root = tree.getroot()
    result = ''
    for cand in root:
        if cand.tag == 'cand':
            mweStr, frequency, features = '', 0, dict()
            for c in cand:
                if c.tag == 'ngram':
                    for w in c:
                        if w.tag == 'w':
                            mweStr += w.attrib['lemma'] + ' '
            result += mweStr.strip() + '\n'
    with open(writingPath, 'w') as csvFile:
        csvFile.write(result)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    # getCandidatesFromXML(cbtScoresXML, rawCandidates)
    getScoresFromXML(cbtScoresXML, cbtScoresCSV, 'cbt')
    # lppLexiconPath = '/Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/mweLEX.txt'
    # createPatternXml(lppLexiconPath, lexiconPatternXmlPath)