from corpus import *
from param import FeatParams


def extract(corpus):
    labels, featureDic = [], []

    for sent in corpus.trainingSents:
        labelsTmp, featuresTmp = extractSent(sent)
        labels.extend(labelsTmp)
        featureDic.extend(featuresTmp)
    return labels, featureDic


def extractSent(sent):
    transition = sent.initialTransition
    labels, features = [], []

    while transition.next:
        # if not transition.next or not transition.next.type:
        #     pass
        if transition.next and transition.next.type:
            labels.append(transition.next.type.value)
            features.append(getFeatures(transition, sent))
            transition = transition.next
    sent.featuresInfo = [labels, features]
    return labels, features


def getFeatures(transition, sent):
    transDic = {}
    configuration = transition.configuration

    # if FeatParams.smartMWTDetection:
    #     if configuration.stack and isinstance(configuration.stack[-1], Token) \
    #             and configuration.stack[-1].getLemma() in Corpus.mwtDictionary:
    #         transDic['isMWT_' + Corpus.mwtDictionary[configuration.stack[-1].getLemma()].lower()] = True
    # TODO return transDic directly in this case
    if FeatParams.useStackLength and len(configuration.stack) > 1:
        transDic['StackLengthIs'] = len(configuration.stack)

    if len(configuration.stack) >= 2:
        stackElements = [configuration.stack[-2], configuration.stack[-1]]
    else:
        stackElements = configuration.stack

    # General linguistic Informations
    if stackElements:
        elemIdx = len(stackElements) - 1
        for elem in stackElements:
            generateLinguisticFeatures(elem, 'S' + str(elemIdx), transDic)
            elemIdx -= 1

    if len(configuration.buffer) > 0:
        if FeatParams.useFirstBufferElement:
            generateLinguisticFeatures(configuration.buffer[0], 'B0', transDic)

        if FeatParams.useSecondBufferElement and len(configuration.buffer) > 1:
            generateLinguisticFeatures(configuration.buffer[1], 'B1', transDic)

    # Bi-Gram Generation
    if FeatParams.useBiGram:
        if len(stackElements) > 1:
            # Generate a Bi-gram S1S0 S0B0 S1B0 S0B1
            generateBiGram(stackElements[-2], stackElements[-1], 'S1S0', transDic)
            if FeatParams.generateS1B1 and len(configuration.buffer) > 1:
                generateBiGram(stackElements[-2], configuration.buffer[1], 'S1B1', transDic)
        if len(stackElements) > 0 and len(configuration.buffer) > 0:
            generateBiGram(stackElements[-1], configuration.buffer[0], 'S0B0', transDic)
            if len(stackElements) > 1:
                generateBiGram(stackElements[-2], configuration.buffer[0], 'S1B0', transDic)
            if len(configuration.buffer) > 1:
                generateBiGram(stackElements[-1], configuration.buffer[1], 'S0B1', transDic)
                if FeatParams.generateS0B2Bigram and len(configuration.buffer) > 2:
                    generateBiGram(stackElements[-1], configuration.buffer[2], 'S0B2', transDic)

    # Tri-Gram Generation
    if FeatParams.useTriGram and len(stackElements) > 1 and len(configuration.buffer) > 0:
        generateTriGram(stackElements[-2], stackElements[-1], configuration.buffer[0], 'S1S0B0', transDic)

    # Syntaxic Informations
    if len(stackElements) > 0 and FeatParams.useSyntax:
        generateSyntaxicFeatures(configuration.stack, configuration.buffer, transDic)

    # Distance information
    if FeatParams.useS0B0Distance and len(configuration.stack) > 0 and len(configuration.buffer) > 0:
        stackTokens = getTokens(configuration.stack[-1])
        transDic['S0B0Distance'] = str(
            sent.tokens.index(configuration.buffer[0]) - sent.tokens.index(stackTokens[-1]))
    if FeatParams.useS0S1Distance and len(configuration.stack) > 1 and isinstance(configuration.stack[-1], Token) \
            and isinstance(configuration.stack[-2], Token):
        transDic['S0S1Distance'] = str(
            sent.tokens.index(configuration.stack[-1]) - sent.tokens.index(configuration.stack[-2]))
    addTransitionHistory(transition, transDic)

    if FeatParams.useLexic and len(configuration.buffer) > 0 and len(configuration.stack) >= 1:
        generateDisconinousFeatures(configuration, sent, transDic)

    enhanceMerge(transition, transDic)

    return transDic


def enhanceMerge(transition, transDic):
    if not FeatParams.enhanceMerge:
        return
    config = transition.configuration
    if transition.type.value != 0 and len(config.buffer) > 0 and len(
            config.stack) > 0 and isinstance(config.stack[-1], Token):
        if isinstance(config.stack[-1], Token) and areInLexic([config.stack[-1], config.buffer[0]]):
            transDic['S0B0InLexic'] = True

        if len(config.buffer) > 1 and areInLexic([config.stack[-1], config.buffer[0], config.buffer[1]]):
            transDic['S0B0B1InLexic'] = True
        if len(config.buffer) > 2 and areInLexic(
                [config.stack[-1], config.buffer[0], config.buffer[1], config.buffer[2]]):
            transDic['S0B0B1B2InLexic'] = True
        if len(config.buffer) > 1 and len(config.stack) > 1 and areInLexic(
                [config.stack[-2], config.stack[-1], config.buffer[1]]):
            transDic['S1S0B1InLexic'] = True

    if len(config.buffer) > 0 and len(config.stack) > 1 and areInLexic(
            [config.stack[-2], config.buffer[0]]) and not areInLexic(
        [config.stack[-1], config.buffer[0]]):
        transDic['S1B0InLexic'] = True
        transDic['S0B0tInLexic'] = False
        if len(config.buffer) > 1 and areInLexic(
                [config.stack[-2], config.buffer[1]]) and not areInLexic(
            [config.stack[-1], config.buffer[1]]):
            transDic['S1B1InLexic'] = True
            transDic['S0B1InLexic'] = False


def generateDisconinousFeatures(configuration, sent, transDic):
    tokens = getTokens([configuration.stack[-1]])
    tokenTxt = getTokenLemmas(tokens)
    for key in Corpus.mweDictionary.keys():
        if tokenTxt in key and tokenTxt != key:
            bufidx = 0
            for bufElem in configuration.buffer[:5]:
                if bufElem.lemma != '' and (
                                (tokenTxt + ' ' + bufElem.lemma) in key or (bufElem.lemma + ' ' + tokenTxt) in key):
                    transDic['S0B' + str(bufidx) + 'ArePartsOfMWE'] = True
                    transDic['S0B' + str(bufidx) + 'ArePartsOfMWEDistance'] = sent.tokens.index(
                        bufElem) - sent.tokens.index(tokens[-1])
                bufidx += 1
            break


def generateLinguisticFeatures(token, label, transDic):
    if isinstance(token, list):
        token = concatenateTokens([token])[0]
    transDic[label + 'Token'] = token.text
    if FeatParams.usePOS and token.posTag is not None and token.posTag.strip() != '':
        transDic[label + 'POS'] = token.posTag
    if FeatParams.useLemma and token.lemma is not None and token.lemma.strip() != '':
        transDic[label + 'Lemma'] = token.lemma
    if not FeatParams.useLemma and not FeatParams.usePOS:
        transDic[label + '_LastThreeLetters'] = token.text[-3:]
        transDic[label + '_LastTwoLetters'] = token.text[-2:]
    if FeatParams.useDictionary and ((token.lemma != '' and token.lemma in Corpus.mweTokenDic.keys())
                                     or token.text in Corpus.mweTokenDic.keys()):
        transDic[label + 'IsInLexic'] = 'true'


def generateSyntaxicFeatures(stack, buffer, dic):
    if stack and isinstance(stack[-1], Token):
        stack0 = stack[-1]
        if int(stack0.dependencyParent) == -1 or int(
                stack0.dependencyParent) == 0 or stack0.dependencyLabel.strip() == '' or not buffer:
            return
        for bElem in buffer:
            if bElem.dependencyParent == stack0.position:
                dic['hasRighDep_' + bElem.dependencyLabel] = 'true'
                dic[stack0.getLemma() + '_hasRighDep_' + bElem.dependencyLabel] = 'true'
                dic[stack0.getLemma() + '_' + bElem.getLemma() + '_hasRighDep_' + bElem.dependencyLabel] = 'true'

        if stack0.dependencyParent > stack0.position:
            for bElem in buffer:
                if bElem.position == stack0.dependencyParent:
                    dic[stack0.lemma + '_isGouvernedBy_' + bElem.getLemma()] = 'true'
                    dic[stack0.lemma + '_isGouvernedBy_' + bElem.getLemma() + '_' + stack0.dependencyLabel] = 'true'
                    break
        if len(stack) > 1:
            stack1 = stack[-2]
            if not isinstance(stack1, Token):
                return
            if stack0.dependencyParent == stack1.position:
                dic['SyntaxicRelation'] = '+' + stack0.dependencyLabel
            elif stack0.position == stack1.dependencyParent:
                dic['SyntaxicRelation'] = '-' + stack1.dependencyLabel


def generateTriGram(token0, token1, token2, label, transDic):
    tokens = concatenateTokens([token0, token1, token2])
    getFeatureInfo(transDic, label + 'Token', tokens, 'ttt')
    getFeatureInfo(transDic, label + 'Lemma', tokens, 'lll')
    getFeatureInfo(transDic, label + 'POS', tokens, 'ppp')
    getFeatureInfo(transDic, label + 'LemmaPOSPOS', tokens, 'lpp')
    getFeatureInfo(transDic, label + 'POSLemmaPOS', tokens, 'plp')
    getFeatureInfo(transDic, label + 'POSPOSLemma', tokens, 'ppl')
    getFeatureInfo(transDic, label + 'LemmaLemmaPOS', tokens, 'llp')
    getFeatureInfo(transDic, label + 'LemmaPOSLemma', tokens, 'lpl')
    getFeatureInfo(transDic, label + 'POSLemmaLemma', tokens, 'pll')


def generateBiGram(token0, token1, label, transDic):
    tokens = concatenateTokens([token0, token1])
    getFeatureInfo(transDic, label + 'Token', tokens, 'tt')
    getFeatureInfo(transDic, label + 'Lemma', tokens, 'll')
    getFeatureInfo(transDic, label + 'POS', tokens, 'pp')
    getFeatureInfo(transDic, label + 'LemmaPOS', tokens, 'lp')
    getFeatureInfo(transDic, label + 'POSLemma', tokens, 'pl')


def concatenateTokens(tokens):
    idx = 0
    tokenDic = {}
    result = []
    for token in tokens:
        if isinstance(token, Token):
            result.append(Token(-1, token.text, token.lemma, token.posTag))
        elif isinstance(token, list):
            tokenDic[idx] = Token(-1, '', '', '')
            for subToken in getTokens(token):
                tokenDic[idx].text += subToken.text + '_'
                tokenDic[idx].lemma += subToken.lemma + '_'
                tokenDic[idx].posTag += subToken.posTag + '_'
            tokenDic[idx].text = tokenDic[idx].text[:-1]
            tokenDic[idx].lemma = tokenDic[idx].lemma[:-1]
            tokenDic[idx].posTag = tokenDic[idx].posTag[:-1]
            result.append(tokenDic[idx])
        idx += 1
    return result


def getFeatureInfo(dic, label, tokens, features):
    feature = ''
    idx = 0
    for token in tokens:
        if features[idx].lower() == 'l':
            if FeatParams.useLemma:
                if token.lemma.strip() != '':
                    feature += token.lemma.strip() + '_'
                else:
                    feature += '*' + '_'
        elif features[idx].lower() == 'p':
            if FeatParams.usePOS:
                if token.posTag.strip() != '':
                    feature += token.posTag.strip() + '_'
                else:
                    feature += '*' + '_'
        elif features[idx].lower() == 't':
            if token.text.strip() != '':
                feature += token.text.strip() + '_'
        idx += 1
    if len(feature) > 0:
        feature = feature[:-1]
        dic[label] = feature

    return ''


def areInLexic(tokensList):
    if getTokenLemmas(tokensList) in Corpus.mweDictionary.keys():
        return True
    return False


def addTransitionHistory(transition, transDic):
    if FeatParams.historyLength1:
        getTransitionHistory(transition, 1, 'TransHistory1', transDic)
    if FeatParams.historyLength2:
        getTransitionHistory(transition, 2, 'TransHistory2', transDic)
    if FeatParams.historyLength3:
        getTransitionHistory(transition, 3, 'TransHistory3', transDic)


def getTransitionHistory(transition, length, label, transDic):
    idx = 0
    history = ''
    transRef = transition
    transition = transition.previous
    while transition is not None and idx < length:
        if transition.type is not None:
            history += str(transition.type.value)
        transition = transition.previous
        idx += 1
    if len(history) == length:
        transDic[label] = history
    transition = transRef
