#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi



for dataset in toy fb15k fb15k-237 wn18 wnrr dbpedia50 dbpedia500 db100k yago3-10; do
  for file in dataset.yaml train.del test.del valid.del valid_without_unseen.del test_without_unseen.del; do
    cmp $BASEDIR/$dataset/$file $BASEDIR/${dataset}_orig/$file
  done
done