from settings import CHECK_FOR_MWE, RIGHT_MERGE
from transitions import *
from transitions import Shift, getMWTTypeFromStr, getTypeFromStr


def parse(corpus):
    for sent in corpus:
        sent.initialTransition = Transition(isInitial=True, sent=sent)
        transition = sent.initialTransition
        while not transition.isTerminal():
            if CHECK_FOR_MWE:
                newTransition = checkMergeAsMWT(transition)
                if newTransition:
                    transition = newTransition
                    continue
            newTransition = checkBlackMerge(transition)
            if newTransition:
                transition = newTransition
                continue
            newTransition = checkReduce(transition)
            if newTransition:
                transition = newTransition
                continue
            shift = Shift(sent=sent)
            shift.apply(transition, sent)
            transition = shift


def checkMergeAsMWT(transition):
    config = transition.configuration
    sent = config.sent

    if config.stack and isinstance(config.stack[-1], Token):
        mwt = config.stack[-1].isMWT()
        if not mwt:
            return None
        type = mwt.type
        markAsMWT = MarkAsMWT(type=getMWTTypeFromStr(type), sent=sent)
        markAsMWT.apply(transition, sent)
        return markAsMWT
    return None


def checkBlackMerge(transition):
    config = transition.configuration
    if len(config.stack) > 1:
        sent = config.sent
        s0Tokens = getTokens(config.stack[-1])
        s1Tokens = getTokens(config.stack[-2])
        # #TODO getParent MWE for WHite merge
        tokens = s1Tokens + s0Tokens
        selectedParents = VMWE.getParents(tokens)
        if selectedParents and len(selectedParents) == 1:
            selectedParent = selectedParents[0]
            selectedParent.parsedByOracle = True

            type = getTypeFromStr(selectedParent.type)
            merge = BlackMerge(type=type, sent=sent)
            merge.apply(transition, sent=sent)
            return merge

        parent = VMWE.haveSameParent(tokens)
        if parent:
            if RIGHT_MERGE and parent.tokens[-1] == tokens[-1]:
                merge = WhiteMerge(sent=sent)
                merge.apply(transition, sent)
                return merge
            elif not RIGHT_MERGE:
                merge = WhiteMerge(sent=sent)
                merge.apply(transition, sent)
                return merge

        selectedParents = VMWE.haveSameParents(tokens)
        if selectedParents and len(selectedParents) == 1:
            if sent.containsEmbedding:
                if selectedParents[0].tokens[-1] == tokens[-1]:
                    merge = WhiteMerge(sent=sent)
                    merge.apply(transition, sent)
                    return merge
            else:
                # merge = WhiteMerge(sent=sent)
                # merge.apply(transition, sent)
                # return merge
                # # @TODO
                if selectedParents[0].tokens[-1] == tokens[-1]:
                    if len(config.stack) > 2:
                        merge = WhiteMerge(sent=sent)
                        merge.apply(transition, sent)
                        return merge

    return None


def checkReduce(parent):
    config = parent.configuration
    stack = config.stack
    sent = config.sent
    reduce = Reduce(sent=sent)

    stackWithTopTokenWitoutParents = stack and isinstance(stack[-1], Token) and (not stack[-1].parentMWEs)
    if stackWithTopTokenWitoutParents:
        reduce.apply(parent, sent)
        return reduce

    empyBufferWithFullStack = not config.buffer and stack
    if empyBufferWithFullStack:
        reduce.apply(parent, sent)
        return reduce

    stackWithMWT = stack and isinstance(stack[-1], list) and len(stack[-1]) == 1 and stack[-1][0].parentMWEs == 1
    if stackWithMWT:
        reduce.apply(parent, sent)
        return reduce

    stackWithSingleListWitOneSharedParentOnly = False
    if stack and isinstance(stack[-1], list):
        tokens = getTokens(stack[-1])
        if len(VMWE.getParents(tokens)) == 1 and not VMWE.getParents(tokens)[0].isEmbedded:
            stackWithSingleListWitOneSharedParentOnly = True

    if stackWithSingleListWitOneSharedParentOnly:
        reduce.apply(parent, sent)
        return reduce

    stackWithTopTokenOfInterleavingMWE = sent.containsInterleaving and stack and isinstance(stack[-1], Token) and (
        stack[-1].parentMWEs and len(stack[-1].parentMWEs) == 1 and stack[-1].parentMWEs[0].isInterleaving)

    if stackWithTopTokenOfInterleavingMWE:
        reduce.apply(parent, sent)
        return reduce
    return None
