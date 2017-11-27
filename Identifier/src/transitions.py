from corpus import VMWE, Token, getTokens
from tranType import TransitionType


class Transition(object):
    def __init__(self, type=None, config=None, previous=None, next=None, isInitial=False, sent=None, confidence=0):

        self.sent = sent
        self.confidence = confidence
        if isInitial:
            self.configuration = Configuration([], sent.tokens, [], sent, self, isInitial=True)
            self.type = None
            sent.initialTransition = self
        else:
            self.configuration = config
        if type:
            self.type = type
        self.previous = previous
        if previous:
            previous.next = self
        self.next = next
        if not self.previous:
            self.id = 0
        else:
            self.id = self.previous.id + 1

            # super(EmbeddingTransition, self).__init__(type, config, previous, next, isInitial, sent)

    def apply(self, parent, sent, parse=False, confidence=0):
        pass

    def getLegalTransDic(self):

        config = self.configuration
        if config and config.legalTrans:
            return config.legalTrans
        transitions = {}
        if len(config.stack) and isinstance(config.stack[-1], Token):
            transitions[TransitionType.MERGE_AS_MWT_ID] = MarkAsMWT(type=TransitionType.MERGE_AS_MWT_ID,
                                                                    sent=self.sent)
            transitions[TransitionType.MERGE_AS_MWT_VPC] = MarkAsMWT(type=TransitionType.MERGE_AS_MWT_VPC,
                                                                     sent=self.sent)
            transitions[TransitionType.MERGE_AS_MWT_LVC] = MarkAsMWT(type=TransitionType.MERGE_AS_MWT_LVC,
                                                                     sent=self.sent)
            transitions[TransitionType.MERGE_AS_MWT_IREFLV] = MarkAsMWT(type=TransitionType.MERGE_AS_MWT_IREFLV,
                                                                        sent=self.sent)
            transitions[TransitionType.MERGE_AS_MWT_OTH] = MarkAsMWT(type=TransitionType.MERGE_AS_MWT_OTH,
                                                                     sent=self.sent)

        if len(config.stack) > 1:
            mergeAsIReflV = BlackMerge(type=TransitionType.MERGE_AS_IREFLV, sent=self.sent)
            transitions[TransitionType.MERGE_AS_IREFLV] = mergeAsIReflV

            mergeAsID = BlackMerge(type=TransitionType.MERGE_AS_ID, sent=self.sent)
            transitions[TransitionType.MERGE_AS_ID] = mergeAsID

            mergeAsLVC = BlackMerge(type=TransitionType.MERGE_AS_LVC, sent=self.sent)
            transitions[TransitionType.MERGE_AS_LVC] = mergeAsLVC

            mergeAsVPC = BlackMerge(type=TransitionType.MERGE_AS_VPC, sent=self.sent)
            transitions[TransitionType.MERGE_AS_VPC] = mergeAsVPC

            mergeAsOTH = BlackMerge(type=TransitionType.MERGE_AS_OTH, sent=self.sent)
            transitions[TransitionType.MERGE_AS_OTH] = mergeAsOTH

            whiteMerge = WhiteMerge(sent=self.sent)
            transitions[TransitionType.WHITE_MERGE] = whiteMerge

        if config.buffer and len(config.buffer):
            transitions[TransitionType.SHIFT] = Shift(sent=self.sent)

        if config.stack and len(config.stack):
            transitions[TransitionType.REDUCE] = Reduce(sent=self.sent)
        config.legalTrans = transitions

        return transitions

    def isTerminal(self):
        if self.configuration.stack or self.configuration.buffer:
            return False
        return True

    def __str__(self):

        configuration = str(self.configuration)
        typeStr = '{0}'.format(self.type.name) if self.type else ''
        typeStr += ' ' * (15 - len(typeStr))

        return '\n\n{0} - {1} : {2}'.format(self.id, typeStr, configuration)


class Shift(Transition):
    def __init__(self, type=TransitionType.SHIFT, config=None, previous=None, next=None, isInitial=False, sent=None,
                 confidence=0):
        super(Shift, self).__init__(type, config, previous, next, isInitial, sent, confidence)
        self.type = TransitionType.SHIFT

    def apply(self, parent, sent, parse=False, confidence=0):
        config = parent.configuration
        lastToken = config.buffer[0]
        newStack = list(config.stack)
        newStack.append(lastToken)
        newConfig = Configuration(newStack, config.buffer[1:], list(config.tokens), sent, self)
        super(Shift, self).__init__(config=newConfig, previous=parent, sent=sent, confidence=confidence)

    def isLegal(self):
        if self.configuration.buffer:
            return True
        return False


class Reduce(Transition):
    def __init__(self, type=TransitionType.REDUCE, config=None, previous=None, next=None, isInitial=False, sent=None,
                 confidence=0):
        super(Reduce, self).__init__(type, config, previous, next, isInitial, sent, confidence)

    def apply(self, parent, sent, parse=False, confidence=0):
        config = parent.configuration
        newBuffer = list(config.buffer)
        newStack = list(config.stack)
        newStack = newStack[:-1]
        newTokens = list(config.tokens)
        newConfig = Configuration(newStack, newBuffer, newTokens, sent, self)
        super(Reduce, self).__init__(config=newConfig, previous=parent, sent=sent, confidence=confidence)

    def isLegal(self):
        if self.configuration.stack:
            return True
        return False


class WhiteMerge(Transition):
    def __init__(self, config=None, previous=None, next=None, isInitial=False, sent=None, confidence=0):
        super(WhiteMerge, self).__init__(TransitionType.WHITE_MERGE, config, previous, next, isInitial, sent,
                                         confidence)

    def apply(self, parent, sent, parse=False, confidence=0):
        config = parent.configuration
        newBuffer = list(config.buffer)
        newStack = list(config.stack)[:-2]
        newStack.append([config.stack[-2], config.stack[-1]])
        newTokens = list(config.tokens)
        newConfig = Configuration(newStack, newBuffer, newTokens, sent, self)

        super(WhiteMerge, self).__init__(config=newConfig, previous=parent, sent=sent, confidence=confidence)

    def isLegal(self):
        if self.configuration.stack and len(self.configuration.stack) > 1:
            return True
        return False


class BlackMerge(Transition):
    def __init__(self, type=None, config=None, previous=None, next=None, isInitial=False, sent=None, confidence=0):
        super(BlackMerge, self).__init__(type, config, previous, next, isInitial, sent, confidence)

    def apply(self, parent, sent, parse=False, confidence=0):
        config = parent.configuration
        newBuffer = list(config.buffer)
        newStack = list(config.stack)[:-2]
        newStack.append([config.stack[-2], config.stack[-1]])
        newTokens = list(config.tokens)
        vMWETokens = getTokens(newStack[-1])
        if len(vMWETokens) > 1:
            if parse:
                vMWEId = len(sent.identifiedVMWEs) + 1
                vMWE = VMWE(vMWEId, vMWETokens[0])
                sent.identifiedVMWEs.append(vMWE)
                vMWE.tokens = vMWETokens
                vMWE.type = getStrFromTransType(self.type)
            else:
                #     vMWEId = len(sent.vMWEs) + 1
                vMWE = VMWE.getParents(vMWETokens)
            # sent.vMWEs.append(vMWE)
            newTokens.append(vMWE)
        elif len(vMWETokens) == 1:
            newTokens.append(vMWETokens[0])

        newConfig = Configuration(newStack, newBuffer, newTokens, sent, self)

        super(BlackMerge, self).__init__(config=newConfig, previous=parent, sent=sent, confidence=confidence)

    def isLegal(self):
        if self.configuration.stack and len(self.configuration.stack) > 1:
            return True
        return False


class MarkAsMWT(BlackMerge):
    def __init__(self, type, config=None, previous=None, next=None, isInitial=False, sent=None, confidence=0):
        super(MarkAsMWT, self).__init__(type, config, previous, next, isInitial, sent, confidence)

    def apply(self, parent, sent, parse=False, confidence=0):
        config = parent.configuration
        newBuffer = list(config.buffer)
        newStack = list(config.stack)[:-1]
        newStack.append([config.stack[-1]])
        newTokens = list(config.tokens)
        vMWETokens = getTokens(newStack[-1])
        if parse:
            vMWEId = len(sent.identifiedVMWEs) + 1
            vMWE = VMWE(vMWEId, vMWETokens[0])
            sent.identifiedVMWEs.append(vMWE)
        else:
            vMWEId = len(sent.vMWEs) + 1
            vMWE = VMWE(vMWEId, vMWETokens[0])
            sent.vMWEs.append(vMWE)
        vMWE.type = getStrFromTransType(self.type)
        newTokens.append(vMWE)
        newConfig = Configuration(newStack, newBuffer, newTokens, sent, self)
        super(MarkAsMWT, self).__init__(config=newConfig, previous=parent, sent=sent, confidence=confidence)

    def isLegal(self):
        if self.configuration.stack:
            return True
        return False


class Configuration:
    def __init__(self, stack, buffer, tokens, sent, transition, isInitial=False, ):

        self.buffer = buffer
        self.stack = stack
        self.tokens = tokens
        self.isInitial = isInitial
        self.isTerminal = self.isTerminal()
        self.sent = sent
        self.transition = transition
        self.legalTrans = {}

    def isTerminal(self):
        if not self.buffer and not self.stack:
            return True
        return False

    def __str__(self):

        stackStr = printStack(self.stack)
        if self.buffer:
            buffStr = '[' + self.buffer[0].text
            if len(self.buffer) > 1:
                buffStr += ', ' + self.buffer[1].text
                if len(self.buffer) > 2:
                    buffStr += ', ' + self.buffer[2].text + ' ,.. '
                else:
                    buffStr += ' ,.. '
            buffStr += ']'
        else:
            buffStr = '[ ]'
        return 'S= ' + stackStr + ' B= ' + buffStr  # + ' ; VMWEs = ' + tokensStr


def initialize(transType, sent, confidence=0):
    if isinstance(transType, int):
        transType = getType(transType)
    if transType == TransitionType.SHIFT:
        return Shift(sent=sent, confidence=confidence)
    if transType == TransitionType.REDUCE:
        return Reduce(sent=sent, confidence=confidence)
    if transType == TransitionType.WHITE_MERGE:
        return WhiteMerge(sent=sent, confidence=confidence)
    if transType in {TransitionType.MERGE_AS_VPC, TransitionType.MERGE_AS_ID, TransitionType.MERGE_AS_IREFLV,
                     TransitionType.MERGE_AS_LVC, TransitionType.MERGE_AS_OTH}:
        return BlackMerge(type=transType, sent=sent, confidence=confidence)
    if transType:
        return MarkAsMWT(type=transType, sent=sent, confidence=confidence)
    return None


def getType(idx):
    for type in TransitionType:
        if type.value == idx:
            return type
    return None


def getTypeFromStr(type):
    if type.lower() == 'vpc':
        return TransitionType.MERGE_AS_VPC
    if type.lower() == 'ireflv':
        return TransitionType.MERGE_AS_IREFLV
    if type.lower() == 'lvc':
        return TransitionType.MERGE_AS_LVC
    if type.lower() == 'id':
        return TransitionType.MERGE_AS_ID
    return TransitionType.MERGE_AS_OTH


def getMWTTypeFromStr(type):
    if type.lower() == 'vpc':
        return TransitionType.MERGE_AS_MWT_VPC
    if type.lower() == 'ireflv':
        return TransitionType.MERGE_AS_MWT_IREFLV
    if type.lower() == 'lvc':
        return TransitionType.MERGE_AS_MWT_LVC
    if type.lower() == 'id':
        return TransitionType.MERGE_AS_MWT_ID
    return TransitionType.MERGE_AS_MWT_OTH


def getStrFromTransType(transType):
    if transType in {TransitionType.MERGE_AS_MWT_VPC, TransitionType.MERGE_AS_VPC}:
        return 'vpc'
    if transType in {TransitionType.MERGE_AS_MWT_IREFLV, TransitionType.MERGE_AS_IREFLV}:
        return 'ireflv'
    if transType in {TransitionType.MERGE_AS_MWT_LVC, TransitionType.MERGE_AS_LVC}:
        return 'lvc'
    if transType in {TransitionType.MERGE_AS_MWT_ID, TransitionType.MERGE_AS_ID}:
        return 'id'
    return 'oth'


def printStack(elemlist):
    result = '['
    for elem in elemlist:
        if isinstance(elem, Token):
            result += elem.text + ', '
        elif isinstance(elem, list):
            result += printStack(elem)
    if result == '[':
        return result + ']  ' + ' ' * (25 - len(result))
    result = result[:-2] + ']  ' + ' ' * (27 - len(result))
    return result
