#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR/data" ]; then
    mkdir "$BASEDIR/data"
fi

# FB15K
if [ ! -d "$BASEDIR/data/FB15k" ]; then
  echo Downloading FB15k
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/FB15k.zip
  # http://web.informatik.uni-mannheim.de/RuleN/data/FB15k.zip
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
else
    echo FB15k already present
fi

# FB15K-237
if [ ! -d "$BASEDIR/data/FB15k-237" ]; then
  echo Downloading FB15k-237
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/FB15k-237.zip
  # http://web.informatik.uni-mannheim.de/RuleN/data/FB15k-237.zip
  unzip FB15k-237.zip
  cd ..
else
    echo FB15k-237 already present
fi

# WN18
if [ ! -d "$BASEDIR/data/WN18" ]; then
  echo Downloading WN18
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/WN18.zip
  # http://web.informatik.uni-mannheim.de/RuleN/data/WN18.zip
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
else
    echo WN18 already present
fi

# WN18RR
if [ ! -d "$BASEDIR/data/WN18RR" ]; then
  echo Downloading WN18RR
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/WN18RR.tar.gz
  tar xvf WN18RR.tar.gz
else
    echo WN18RR already present
fi


# DBpedia50
if [ ! -d "$BASEDIR/data/dbpedia50" ]; then
  echo Downloading dbpedia50
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/dbpedia50.tar.gz 
  tar xvf dbpedia50.tar.gz
else
    echo dbpedia50 already present
fi

# DBpedia500
if [ ! -d "$BASEDIR/data/dbpedia500" ]; then
  echo Downloading dbpedia500
  cd $BASEDIR/data
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/dbpedia500.tar.gz 
  tar xvf dbpedia500.tar.gz
else
    echo dbpedia500 already present
fi

# DBpedia100
if [ ! -d "$BASEDIR/data/DB100K" ]; then
  echo Downloading DB100K
  cd $BASEDIR/data
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
else
    echo DB100K already present
fi

# YAGO3-10
if [ ! -d "$BASEDIR/data/YAGO3-10" ]; then
  echo Downloading YAGO3-10
  curl -O https://www.wim.uni-mannheim.de/fileadmin/lehrstuehle/pi1/pi1/BK-EMB/YAGO3-10.tar.gz
  tar xvf YAGO3-10.tar.gz
else
    echo YAGO3-10 already present
fi
