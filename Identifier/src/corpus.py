import collections
import os
from random import shuffle

import settings
from param import *
from tranType import TransitionType

projectPath = '/Users/halsaied/PycharmProjects/LePetitPrince/'

# trainConllFile = projectPath + 'Corpora/CBT/conll.txt'
# trainMweFile = projectPath + 'Corpora/CBT/mwes.txt'

corpusPath = os.path.join(projectPath, 'Corpora/CBT')

conllFolder = os.path.join(corpusPath, 'Divided-conll')
mweFolder = os.path.join(corpusPath, 'Divided-MWE')

trainConllFile = projectPath + 'Corpora/LPP/conll.txt'
trainMweFile = projectPath + 'Corpora/LPP/mwe.txt'

testConllFile = projectPath + 'Corpora/LPP/conll.txt'
testMweFile = projectPath + 'Corpora/LPP/mwe.txt'


def readConlluFile(conlluFile):
    sentences = []
    with open(conlluFile) as corpusFile:
        sent, senIdx, sentId = None, 0, ''
        lineNum, missedUnTag, missedExTag = 0, 0, 0
        for line in corpusFile:
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('# sentid:'):
                sentId = line.split('# sentid:')[1].strip()
            elif line.startswith('# sentence-text:'):
                continue

            elif line.startswith('1\t'):
                if sentId.strip():
                    sent = Sentence(senIdx, sentid=sentId)
                else:
                    sent = Sentence(senIdx)
                senIdx += 1
                sentences.append(sent)

            if not line.startswith('#'):
                lineParts = line.split('\t')

                if len(lineParts) != 10 or '-' in lineParts[0]:
                    continue

                lineNum += 1
                if lineParts[3] == '_':
                    missedUnTag += 1
                if lineParts[4] == '_':
                    missedExTag += 1

                morpho = ''
                if lineParts[5] != '_':
                    morpho = lineParts[5].split('|')
                if lineParts[6] != '_':
                    token = Token(lineParts[0], lineParts[1].lower(), lemma=lineParts[2],
                                  abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                  dependencyParent=int(lineParts[6]),
                                  dependencyLabel=lineParts[7])
                else:
                    token = Token(lineParts[0], lineParts[1].lower(), lemma=lineParts[2],
                                  abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                  dependencyLabel=lineParts[7])
                # if settings.CORPUS_USE_UNIVERSAL_POS_TAGS:
                #     token.posTag = lineParts[3]
                # else:
                if lineParts[4] != '_':
                    token.posTag = lineParts[4]
                else:
                    token.posTag = lineParts[3]
                # Associate the token with the sentence
                sent.tokens.append(token)
                sent.text += token.text + ' '
    return sentences


def readSentences(mweFile):
    sentences = []
    sentNum, mweNum = 0, 0
    with open(mweFile) as corpusFile:
        # Read the corpus file
        lines = corpusFile.readlines()
        sent = None
        senIdx = 1
        for line in lines:
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('1\t'):
                # sentId = line.split('# sentid:')[1]
                if sent:
                    # Represent the sentence as a sequece of tokens and POS tags
                    sent.setTextandPOS()
                    # if not forTest:
                    sent.recognizeEmbedded()
                    sent.recognizeInterleavingVMWEs()
                sent = Sentence(senIdx)
                senIdx += 1
                sentences.append(sent)

            elif line.startswith('# sentence-text:'):
                if len(line.split(':')) > 1:
                    sent.text = line.split('# sentence-text:')[1]

            lineParts = line.split('\t')

            # Empty line or lines of the form: "8-9	can't	_	_"
            if len(lineParts) != 4 or '-' in lineParts[0]:
                continue
            token = Token(lineParts[0], lineParts[1])
            # Trait the MWE
            # if not forTest and lineParts[3] != '_':
            if lineParts[3] != '_':
                vMWEids = lineParts[3].split(';')
                for vMWEid in vMWEids:
                    idx = int(vMWEid.split(':')[0])
                    # New MWE captured
                    if idx not in sent.getWMWEIds():
                        type = str(vMWEid.split(':')[1])
                        vMWE = VMWE(idx, token, type)
                        mweNum += 1
                        sent.vMWEs.append(vMWE)
                    # Another token of an under-processing MWE
                    else:
                        vMWE = sent.getVMWE(idx)
                        if vMWE is not None:
                            vMWE.addToken(token)
                    # associate the token with the MWE
                    token.setParent(vMWE)
            # Associate the token with the sentence
            sent.tokens.append(token)
        sentNum = len(sentences)
        return sentences, sentNum, mweNum


def readMweFile(mweFile, sentences):
    mweNum = 0
    with open(mweFile) as corpusFile:
        # Read the corpus file
        lines = corpusFile.readlines()
        noSentToAssign = False
        sentIdx = 0
        for line in lines:
            if line == '\n' or line.startswith('# sentence-text:') or (
                        line.startswith('# sentid:') and noSentToAssign):
                continue
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('1\t'):
                sent = sentences[sentIdx]
                sentIdx += 1
            lineParts = line.split('\t')
            if '-' in lineParts[0]:
                continue
            if lineParts and len(lineParts) == 4 and lineParts[3] != '_':

                token = sent.tokens[int(lineParts[0]) - 1]
                vMWEids = lineParts[3].split(';')
                for vMWEid in vMWEids:
                    idx = int(vMWEid.split(':')[0])
                    # New MWE captured
                    if idx not in sent.getWMWEIds():
                        if len(vMWEid.split(':')) > 1:
                            type = str(vMWEid.split(':')[1])
                            vMWE = VMWE(idx, token, type)
                        else:
                            vMWE = VMWE(idx, token)
                        mweNum += 1
                        sent.vMWEs.append(vMWE)
                    # Another token of an under-processing MWE
                    else:
                        vMWE = sent.getVMWE(idx)
                        if vMWE:
                            vMWE.addToken(token)
                    # associate the token with the MWE
                    token.setParent(vMWE)

    return mweNum


def getVMWESents(sents, num):
    result, idx = [], 0
    if settings.CORPUS_SHUFFLE:
        shuffle(sents)
    for sent in sents:
        if sent.vMWEs:
            result.append(sent)
            idx += 1
        if idx >= num:
            return result
    raise


def getTokens(elemlist):
    if isinstance(elemlist, Token):
        return [elemlist]
    if isinstance(elemlist, collections.Iterable):
        result = []
        for elem in elemlist:
            if isinstance(elem, Token):
                result.append(elem)
            elif isinstance(elem, list):
                result.extend(getTokens(elem))
        return result
    return [elemlist]


def getTokenText(tokens):
    text = ''
    tokens = getTokens(tokens)
    for token in tokens:
            text += token.text + ' '
    return text.strip()

def getTokenLemmas(tokens):
    text = ''
    tokens = getTokens(tokens)
    for token in tokens:
        if token.lemma != '':
            text += token.lemma + ' '
        else:
            text += token.text + ' '
    return text.strip()


def getMWEDic(sents):
    mweDictionary, mweTokenDictionary = {}, {}
    for sent in sents:
        for mwe in sent.vMWEs:
            lemmaString = mwe.getLemmaString()
            if lemmaString in mweDictionary:
                mweDictionary[lemmaString] += 1
                for token in mwe.tokens:
                    if token.lemma.strip() != '':
                        mweTokenDictionary[token.lemma] = 1
                    else:
                        mweTokenDictionary[token.text] = 1
            else:
                mweDictionary[lemmaString] = 1
                for token in mwe.tokens:
                    if token.lemma.strip() != '':
                        mweTokenDictionary[token.lemma] = 1
                    else:
                        mweTokenDictionary[token.text] = 1
    if FeatParams.usePreciseDictionary:
        for key1 in mweDictionary.keys():
            for key2 in mweDictionary.keys():
                if key1 != key2:
                    if key1 in key2:
                        mweDictionary.pop(key1, None)
                    elif key2 in key1:
                        mweDictionary.pop(key2, None)
    return mweDictionary, mweTokenDictionary


class Corpus:
    """
        a class used to encapsulate all the information of the corpus
    """
    mweTokenDic, mweDictionary = {}, {}

    def __init__(self, multipleFile=False):

        self.trainingSents = []
        if multipleFile:
            idxxx = 0
            for conllFile in os.listdir(conllFolder):
                if not conllFile.endswith('.txt'):
                    continue
                idx = conllFile.split('.')[0][5:]
                sents = readConlluFile(os.path.join(conllFolder, conllFile))
                mweFileP = os.path.join(mweFolder, 'mwe' + str(idx) + '.txt')
                readMweFile(mweFileP, sents)
                for sent in sents:
                    sent.recognizeEmbedded()
                    sent.recognizeInterleavingVMWEs()
                    sent.recognizeContinouosandSingleVMWEs()
                self.trainingSents.extend(sents)
                if idxxx == 3:
                    break
                idxxx += 1
        else:
            self.trainingSents = readConlluFile(trainConllFile)
            readMweFile(trainMweFile, self.trainingSents)
            for sent in self.trainingSents:
                sent.recognizeEmbedded()
                sent.recognizeInterleavingVMWEs()
                sent.recognizeContinouosandSingleVMWEs()

        self.testingSents = readConlluFile(testConllFile)
        readMweFile(testMweFile, self.testingSents)

        # Sorting parents to get the direct parent on the top of parentVMWEs list
        for sent in self.trainingSents:
            if sent.vMWEs and sent.containsEmbedding:
                for token in sent.tokens:
                    if token.parentMWEs and len(token.parentMWEs) > 1:
                        token.parentMWEs = sorted(token.parentMWEs, key=lambda mwe: (len(mwe)))
        Corpus.mweDictionary, Corpus.mweTokenDic = getMWEDic(self.trainingSents)

    def __iter__(self):
        for sent in self.trainingSents:
            yield sent


class Sentence:
    """
       a class used to encapsulate all the information of a sentence
    """

    def __init__(self, idx, sentid=''):

        self.sentid = sentid
        self.id = idx
        self.text = ''
        self.tokens = []
        self.vMWEs = []
        self.identifiedVMWEs = []
        self.initialTransition = None
        self.containsEmbedding = False
        self.containsInterleaving = False
        self.containsDistributedEmbedding = False

    def getWMWEIds(self):
        result = []
        for vMWE in self.vMWEs:
            result.append(vMWE.getId())
        return result

    def getVMWE(self, idx):

        for vMWE in self.vMWEs:
            if vMWE.getId() == int(idx):
                return vMWE
        return None

    def setTextandPOS(self):

        tokensTextList = []
        for token in self.tokens:
            self.text += token.text + ' '
            tokensTextList.append(token.text)
        self.text = self.text.strip()

    def recognizeEmbedded(self, recognizeIdentified=False):
        if recognizeIdentified:
            vmws = self.identifiedVMWEs
        else:
            vmws = self.vMWEs

        if len(vmws) <= 1:
            return
        for vMwe1 in vmws:
            if vMwe1.isEmbedded:
                continue
            for vMwe2 in vmws:
                if vMwe1 is not vMwe2 and len(vMwe1.tokens) < len(vMwe2.tokens):
                    if vMwe1.getString() in vMwe2.getString():
                        vMwe1.isEmbedded = True
                        if not recognizeIdentified:
                            self.containsEmbedding = True
                    else:
                        isEmbedded = True
                        vMwe2Lemma = vMwe2.getLemmaString()
                        for token in vMwe1.tokens:
                            if token.getLemma() not in vMwe2Lemma:
                                isEmbedded = False
                                break
                        if isEmbedded:
                            vMwe1.isDistributedEmbedding = True
                            vMwe1.isEmbedded = True
                            if not recognizeIdentified:
                                self.containsDistributedEmbedding = True
                                self.containsEmbedding = True
        if not recognizeIdentified:
            self.getDirectParents()

    def recognizeContinouosandSingleVMWEs(self):
        singleWordExp, continousExp = 0, 0
        for mwe in self.vMWEs:
            if len(mwe.tokens) == 1:
                mwe.isSingleWordExp = True
                mwe.isContinousExp = True
                singleWordExp += 1
                continousExp += 1
            else:
                if self.isContinousMwe(mwe):
                    continousExp += 1
        return singleWordExp, continousExp

    def isContinousMwe(self, mwe):
        idxs = []
        for token in mwe.tokens:
            idxs.append(self.tokens.index(token))
        mwe.isContinousExp = True
        for i in xrange(min(idxs), max(idxs)):
            if i not in idxs:
                mwe.isContinousExp = False
        return mwe.isContinousExp

    def recognizeInterleavingVMWEs(self):
        if len(self.vMWEs) <= 1:
            return 0
        result = 0
        for vmwe in self.vMWEs:
            if vmwe.isEmbedded or vmwe.isInterleaving:
                continue
            for token in vmwe.tokens:
                if len(token.parentMWEs) > 1:
                    for parent in token.parentMWEs:
                        if parent is not vmwe:
                            if parent.isEmbedded or parent.isInterleaving:
                                continue
                            if len(parent.tokens) <= len(vmwe.tokens):
                                parent.isInterleaving = True
                            else:
                                vmwe.isInterleaving = True
                            self.containsInterleaving = True
                            result += 1
        return result

    def getDirectParents(self):
        for token in self.tokens:
            token.getDirectParent()

    def getConfidence(self):
        if self.identifiedVMWEs:
            print self.text
        for mwe in self.identifiedVMWEs:
            print 'MWE: ', getTokenLemmas(mwe.tokens)

            if getTokenLemmas(mwe.tokens) == 'good for':
                pass
            if len(mwe) == 1:
                print 'len = 1!'
                mwe.confidence = 0
                continue
            trans = self.initialTransition.next
            implicatedTransNum, singleLegalTransNum, confidence = (len(mwe) - 1) * 2, 0, 0
            print 'implicatedTransNum : ', implicatedTransNum
            while trans.next:
                if trans.type and trans.type in {TransitionType.MERGE_AS_OTH,
                                                 TransitionType.MERGE_AS_ID,
                                                 TransitionType.MERGE_AS_VPC,
                                                 TransitionType.MERGE_AS_IREFLV,
                                                 TransitionType.MERGE_AS_LVC}:
                    if getTokenLemmas(mwe.tokens) == getTokenLemmas(trans.configuration.stack[-1]):
                        implicatedTrans = [trans]
                        print str(trans.type)[len('TransitionType:'):]
                        print trans.confidence
                        confidence = trans.confidence
                        capturedTransNum = 1
                        trans = trans.previous
                        while trans.previous:
                            if trans.type == TransitionType.REDUCE:
                                if trans.previous and trans.previous.previous:
                                    trans = trans.previous.previous
                                elif trans.previous:
                                    trans = trans.previous
                            else:
                                print str(trans.type)[len('TransitionType:'):]
                                print trans.confidence
                                implicatedTrans.append(trans)
                                confidence += trans.confidence
                                if confidence == 0 or confidence == 1:
                                    singleLegalTransNum += 1
                                capturedTransNum += 1
                                trans = trans.previous
                            if implicatedTransNum == capturedTransNum:
                                break
                        break
                trans = trans.next
            conf = round(float(confidence / (implicatedTransNum - singleLegalTransNum)), 3)
            if conf > 1:
                pass
            mwe.confidence = conf
            print 'mwe.confidence', mwe.confidence

    def __str__(self):

        vMWEText, identifiedMWE = '', ''
        for vMWE in self.vMWEs:
            vMWEText += '\n' + str(vMWE) + '\n'
        if len(self.identifiedVMWEs) > 0:
            identifiedMWE = '### Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += '\n' + str(mwe) + '\n'

        transStr = ''
        trans = self.initialTransition
        while trans:
            transStr += str(trans)
            trans = trans.next
        if self.vMWEs:
            text = ''
            for token in self.tokens:
                if token.parentMWEs is not None and len(token.parentMWEs) > 0:
                    text += '**' + token.text + '**' + ' '
                else:
                    text += token.text + ' '
        else:
            text = self.text
        return '{0} - {1} \n {2} \nMWEs:\n{3}\nIdentified MWEs:\n{4}\nTransitions:\n{5}'.format(self.id, self.sentid,
                                                                                                text, vMWEText,
                                                                                                identifiedMWE, transStr)

    def __iter__(self):
        for vmwe in self.vMWEs:
            yield vmwe


class VMWE:
    """
        A class used to encapsulate the information of a verbal multi-word expression
    """

    def __init__(self, idx, token=None, type='', isEmbedded=False, isInterleaving=False, isDistributedEmbedding=False):
        self.id = int(idx)
        self.tokens = []
        if token:
            self.tokens.append(token)
        self.type = type
        self.isEmbedded = isEmbedded
        self.isDistributedEmbedding = isDistributedEmbedding
        self.isInterleaving = isInterleaving
        self.directParent = None
        self.parsedByOracle = False
        self.confidence = 0

    def getId(self):
        return self.id

    def addToken(self, token):
        self.tokens.append(token)

    @staticmethod
    def getVMWENumber(tokens):
        result = 0
        for token in tokens:
            if isinstance(token, VMWE):
                result += 1
        return result

    @staticmethod
    def haveSameParents(tokens):
        # Do they have all a parent?
        for token in tokens:
            if not token.parentMWEs:
                return None
        # Get all parents of tokens
        parents = set()
        for token in tokens:
            for parent in token.parentMWEs:
                parents.add(parent)
        if len(parents) == 1:
            return list(parents)

        selectedParents = list(parents)
        for parent in parents:
            for token in tokens:
                if parent not in token.parentMWEs:
                    if parent in selectedParents:
                        selectedParents.remove(parent)

        for parent in list(selectedParents):
            if parent.isInterleaving or parent.isDistributedEmbedding:
                selectedParents.remove(parent)

        if len(selectedParents) > 1:
            selectedParents = sorted(selectedParents, key=lambda mwe: (len(mwe)))
        return selectedParents

    @staticmethod
    def haveSameParent(tokens):
        # Do they have all a parent?
        for token in tokens:
            if not token.parentMWEs:
                return None
        # Get all parents of tokens
        parents = set()
        for token in tokens:
            for parent in token.parentMWEs:
                if not parent.parsedByOracle:
                    parents.add(parent)
        if len(parents) == 1:
            return list(parents)[0]
        return None

    @staticmethod
    def getParents(tokens, type=None):
        if len(tokens) == 1:
            if tokens[0].parentMWEs:
                for vmwe in tokens[0].parentMWEs:
                    if len(vmwe.tokens) == 1:  # and vmwe.type.lower() != type:
                        if type is not None:
                            if vmwe.type.lower() == type.lower():
                                return [vmwe]
                            else:
                                return None
                        else:
                            return [vmwe]

        # Do they have all a parent?
        for token in tokens:
            if not token.parentMWEs:
                return None

        # Get all parents of tokens
        parents = set()
        for token in tokens:
            for parent in token.parentMWEs:
                parents.add(parent)
        selectedParents = list(parents)
        for parent in parents:
            if len(parent.tokens) != len(tokens):
                if parent in selectedParents:
                    selectedParents.remove(parent)
                continue
            for token in tokens:
                if parent not in token.parentMWEs:
                    if parent in selectedParents:
                        selectedParents.remove(parent)
        for parent in list(selectedParents):
            if parent.isInterleaving or parent.isDistributedEmbedding:
                selectedParents.remove(parent)
        if type is not None:
            for parent in list(selectedParents):
                if parent.type.lower() != type:
                    selectedParents.remove(parent)
        return selectedParents

    def __str__(self):
        tokensStr = ''
        for token in self.tokens:
            tokensStr += token.text + ' '
        tokensStr = tokensStr.strip()
        isEmbedded = self.isEmbedded or self.isDistributedEmbedding
        isEmbeddedStr = ': Embedded' if isEmbedded else ''
        confidence = 'Confid = {0}'.format(self.confidence) + '\n'
        isInterleavingStr = ': Interleaving' if self.isInterleaving else ''

        return '{0}- {1} : {2} {3} {4} {5} '.format(self.id, self.type, tokensStr, isEmbeddedStr, isInterleavingStr,
                                                    confidence)

    def __iter__(self):
        for t in self.tokens:
            yield t

    def getString(self):
        result = ''
        for token in self.tokens:
            result += token.text + ' '
        return result[:-1].lower()

    def getLemmaString(self):
        result = ''
        for token in self.tokens:
            if token.lemma.strip() != '':
                result += token.lemma + ' '
            else:
                result += token.text + ' '
        return result[:-1].lower()

    def In(self, vmwes):

        for vmwe in vmwes:
            if vmwe.getString() == self.getString():
                return True

        return False

    def __eq__(self, other):
        if not isinstance(other, VMWE):
            raise TypeError()
        if self.getLemmaString() == other.getLemmaString():
            return True
        return False

    def __hash__(self):
        return hash(self.getLemmaString())

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, vmwe):
        if not isinstance(vmwe, VMWE):
            raise TypeError()
        if vmwe is self or vmwe.getLemmaString() == self.getLemmaString():
            return False
        if vmwe.getLemmaString() in self.getLemmaString():
            return True
        for token in vmwe.tokens:
            if token.getLemma() not in self.getLemmaString():
                return False
        return True


class Token:
    """
        a class used to encapsulate all the information of a sentence tokens
    """

    def __init__(self, position, txt, lemma='', posTag='', abstractPosTag='', morphologicalInfo=None,
                 dependencyParent=-1,
                 dependencyLabel=''):
        self.position = int(position)
        self.text = txt
        self.lemma = lemma
        self.abstractPosTag = abstractPosTag
        self.posTag = posTag
        if not morphologicalInfo:
            self.morphologicalInfo = []
        else:
            self.morphologicalInfo = morphologicalInfo
        self.dependencyParent = dependencyParent
        self.dependencyLabel = dependencyLabel
        self.parentMWEs = []
        self.directParent = None

    def setParent(self, vMWE):
        self.parentMWEs.append(vMWE)

    def getLemma(self):
        if self.lemma:
            return self.lemma.strip()
        return self.text.strip()

    def getDirectParent(self):
        self.directParent = None
        if self.parentMWEs:
            if len(self.parentMWEs) == 1:
                if not self.parentMWEs[0].isInterleaving:
                    self.directParent = self.parentMWEs[0]
            else:
                parents = sorted(self.parentMWEs,
                                 key=lambda mwe: (mwe.isInterleaving, mwe.isEmbedded, len(mwe)),
                                 reverse=True)
                for parent in parents:
                    if not parent.isInterleaving:
                        self.directParent = parent
                        break
        return self.directParent

    def In(self, vmwe):
        for token in vmwe.tokens:
            if token.text.lower() == self.text.lower() and token.position == self.position:
                return True
        return False

    def isMWT(self):
        if self.parentMWEs:
            for vmw in self.parentMWEs:
                if len(vmw.tokens) == 1:
                    return vmw
        return None

    def __str__(self):
        parentTxt = ''
        if len(self.parentMWEs) != 0:
            for parent in self.parentMWEs:
                parentTxt += str(parent) + '\n'

        return str(self.position) + ' : ' + self.text + ' : ' + self.posTag + '\n' + 'parent VMWEs\n' + parentTxt
