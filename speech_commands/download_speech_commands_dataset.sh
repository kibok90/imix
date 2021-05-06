#!/usr/bin/env sh
set -e

FILE_NAME=speech_commands_v0.01.tar.gz
URL=http://download.tensorflow.org/data/$FILE_NAME
DATASET_FOLDER=data/speech_commands

echo "downloading $URL...\n"
wget -O $DATASET_FOLDER/$FILE_NAME $URL

echo "extracting $FILE_NAME..."
TEMP_DIRECTORY=$DATASET_FOLDER/audio
mkdir -p $TEMP_DIRECTORY
tar -xzf $DATASET_FOLDER/$FILE_NAME -C $TEMP_DIRECTORY
rm $DATASET_FOLDER/$FILE_NAME

echo "splitting the dataset into train, validation and test sets..."
python speech_commands/split_dataset.py $DATASET_FOLDER

echo "done"
