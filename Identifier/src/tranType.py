from enum import Enum

class TransitionType(Enum):
    SHIFT = 0
    REDUCE = 1
    WHITE_MERGE = 2
    MERGE_AS_LVC = 3
    MERGE_AS_VPC = 4
    MERGE_AS_IREFLV = 5
    MERGE_AS_ID = 6
    MERGE_AS_OTH = 7
    MERGE_AS_MWT_VPC = 8
    MERGE_AS_MWT_IREFLV = 9
    MERGE_AS_MWT_ID = 10
    MERGE_AS_MWT_LVC = 11
    MERGE_AS_MWT_OTH = 12