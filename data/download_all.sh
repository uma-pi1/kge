#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi

# FB15K
if [ ! -d "$BASEDIR/FB15k" ]; then
  echo Downloading FB15k
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/FB15k.zip
  unzip FB15k.zip
  cd FB15k
  case "$(uname -s)" in
      CYGWIN*|MINGW32*|MSYS*)
          cmd.exe /c mklink train.txt freebase_mtr100_mte100-train.txt
          cmd.exe /c mklink valid.txt freebase_mtr100_mte100-valid.txt
          cmd.exe /c mklink test.txt freebase_mtr100_mte100-test.txt
          ;;
      *)
          ln -s freebase_mtr100_mte100-train.txt train.txt
          ln -s freebase_mtr100_mte100-valid.txt valid.txt
          ln -s freebase_mtr100_mte100-test.txt test.txt
          ;;
  esac
  cd ..
  python preprocess.py --folder FB15k
else
    echo FB15k already present
fi

# FB15K-237
if [ ! -d "$BASEDIR/FB15k-237" ]; then
  echo Downloading FB15k-237
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/FB15k-237.zip
  unzip FB15k-237.zip
  python preprocess.py --folder FB15k-237
else
    echo FB15k-237 already present
fi

# WN18
if [ ! -d "$BASEDIR/WN18" ]; then
  echo Downloading WN18
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/WN18.zip
  unzip WN18.zip
  cd WN18
  case "$(uname -s)" in
      CYGWIN*|MINGW32*|MSYS*)
          cmd.exe /c mklink train.txt wordnet-mlj12-train.txt
          cmd.exe /c mklink valid.txt wordnet-mlj12-valid.txt
          cmd.exe /c mklink test.txt wordnet-mlj12-test.txt
          ;;
      *)
          ln -s wordnet-mlj12-train.txt train.txt
          ln -s wordnet-mlj12-valid.txt valid.txt
          ln -s wordnet-mlj12-test.txt test.txt
          ;;
  esac
  cd ..
  python preprocess.py --folder WN18
else
    echo WN18 already present
fi

# WN18RR
if [ ! -d "$BASEDIR/WN18RR" ]; then
  echo Downloading WN18RR
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/WN18RR.tar.gz
  tar xvf WN18RR.tar.gz
  python preprocess.py --folder WN18RR
else
    echo WN18RR already present
fi


# DBpedia50
if [ ! -d "$BASEDIR/dbpedia50" ]; then
  echo Downloading dbpedia50
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/dbpedia50.tar.gz 
  tar xvf dbpedia50.tar.gz
  python preprocess.py --folder dbpedia50
else
    echo dbpedia50 already present
fi

# DBpedia500
if [ ! -d "$BASEDIR/dbpedia500" ]; then
  echo Downloading dbpedia500
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/dbpedia500.tar.gz 
  tar xvf dbpedia500.tar.gz
  python preprocess.py --folder dbpedia500
else
    echo dbpedia500 already present
fi

# DBpedia100
if [ ! -d "$BASEDIR/DB100K" ]; then
  echo Downloading DB100K
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/DB100K.tar.gz 
  tar xvf DB100K.tar.gz
  cd DB100K
  case "$(uname -s)" in
      CYGWIN*|MINGW32*|MSYS*)
          cmd.exe /c mklink train.txt _train.txt
          cmd.exe /c mklink valid.txt _valid.txt
          cmd.exe /c mklink test.txt _test.txt
          ;;
      *)
          ln -s _train.txt train.txt
          ln -s _valid.txt valid.txt
          ln -s _test.txt test.txt
          ;;
  esac
  cd ..
  python preprocess.py --folder DB100K
else
    echo DB100K already present
fi

# YAGO3-10
if [ ! -d "$BASEDIR/YAGO3-10" ]; then
  echo Downloading YAGO3-10
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/YAGO3-10.tar.gz
  tar xvf YAGO3-10.tar.gz
  python preprocess.py --folder YAGO3-10
else
    echo YAGO3-10 already present
fi

# FB1K (testing dataset)
if [ ! -d "$BASEDIR/FB1K" ]; then
  echo Downloading FB1K
  cd $BASEDIR
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/FB1K.tar.gz
  tar xvf FB1K.tar.gz
  python preprocess.py --folder FB1K
else
    echo FB1K already present
fi

