#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi


# toy (testing dataset)
if [ ! -d "$BASEDIR/toy" ]; then
    echo Downloading toy
    cd $BASEDIR
    curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/toy.tar.gz
    tar xvf toy.tar.gz
    python preprocess.py toy
else
    echo toy already present
fi


# fb15k
if [ ! -d "$BASEDIR/fb15k" ]; then
  echo Downloading fb15k
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/fb15k.tar.gz
  tar xvf fb15k.tar.gz
  cd fb15k
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
  python preprocess.py fb15k
else
    echo fb15k already present
fi

# fb15k-237
if [ ! -d "$BASEDIR/fb15k-237" ]; then
  echo Downloading fb15k-237
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/fb15k-237.tar.gz
  tar xvf fb15k-237.tar.gz
  python preprocess.py fb15k-237
else
    echo fb15k-237 already present
fi

# wn18
if [ ! -d "$BASEDIR/wn18" ]; then
  echo Downloading wn18
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/wn18.tar.gz
  tar xvf wn18.tar.gz
  cd wn18
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
  python preprocess.py wn18
else
    echo wn18 already present
fi

# wnrr
if [ ! -d "$BASEDIR/wnrr" ]; then
  echo Downloading wnrr
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/wnrr.tar.gz
  tar xvf wnrr.tar.gz
  python preprocess.py wnrr
else
    echo wnrr already present
fi


# dbpedia50
if [ ! -d "$BASEDIR/dbpedia50" ]; then
  echo Downloading dbpedia50
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/dbpedia50.tar.gz
  tar xvf dbpedia50.tar.gz
  python preprocess.py dbpedia50
else
    echo dbpedia50 already present
fi

# dbpedia500
if [ ! -d "$BASEDIR/dbpedia500" ]; then
  echo Downloading dbpedia500
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/dbpedia500.tar.gz
  tar xvf dbpedia500.tar.gz
  python preprocess.py dbpedia500 --order_sop
else
    echo dbpedia500 already present
fi

# db100k
if [ ! -d "$BASEDIR/db100k" ]; then
  echo Downloading db100k
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/db100k.tar.gz
  tar xvf db100k.tar.gz
  cd db100k
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
  python preprocess.py db100k
else
    echo db100k already present
fi

# yago3-10
if [ ! -d "$BASEDIR/yago3-10" ]; then
  echo Downloading yago3-10
  cd $BASEDIR
  curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/yago3-10.tar.gz
  tar xvf yago3-10.tar.gz
  python preprocess.py yago3-10
else
    echo yago3-10 already present
fi
