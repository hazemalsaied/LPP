#!/bin/bash

cd /Users/halsaied/PycharmProjects/LePetitPrince/mwetoolkit

bin/index.py -v -i indexCBT/cbt /Users/halsaied/PycharmProjects/LePetitPrince/Corpora/CBT/conll-without-pos.txt

bin/candidates.py -p LPP-patterns.xml -S -v --patterns-from XML --corpus-from=BinaryIndex indexCBT/cbt.info > CBT-scores/candidates.xml

bin/counter.py -v -g -i indexCBT/cbt.info CBT-scores/candidates.xml > CBT-scores/candidate-counts.xml

bin/wc.py CBT-scores/candidates.xml

bin/feat_association.py -v  CBT-scores/candidate-counts.xml > CBT-scores/candidates-features.xml