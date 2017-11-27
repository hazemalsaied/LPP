#!/bin/bash

cd /Users/halsaied/PycharmProjects/LePetitPrince/mwetoolkit

bin/index.py -v -i indexLPP/lpp /Users/halsaied/PycharmProjects/LePetitPrince/Corpora/LPP/conll.txt

bin/candidates.py -p LPP-patterns.xml -S -v --patterns-from XML --corpus-from=BinaryIndex indexLPP/lpp.info > LPP-scores/candidates.xml

bin/counter.py -v -i indexLPP/lpp.info LPP-scores/candidates.xml > LPP-scores/candidate-counts.xml

bin/wc.py LPP-scores/candidates.xml

bin/feat_association.py -v LPP-scores/candidate-counts.xml > LPP-scores/candidates-features.xml