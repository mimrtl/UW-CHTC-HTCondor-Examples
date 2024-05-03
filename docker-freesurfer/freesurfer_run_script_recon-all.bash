#!/bin/bash

# get parameters
INPUT_FILE="$1"
echo "MYINFO: Input file is $INPUT_FILE"
OUTPUT_FILE="$2"
echo "MYINFO: Output file is $OUTPUT_FILE"

# input files are in pwd
INITIAL_DIR=$( pwd )
echo "MYINFO: Initial dir is $INITIAL_DIR"

# set environment variable to license file
export FS_LICENSE=$INITIAL_DIR/license.txt

# make sure data is written to the inital directory
export SUBJECTS_DIR=$INITIAL_DIR

# run recon-all
echo "MYINFO: Starting freesurfer"
recon-all -s sub-101 -i $INITIAL_DIR/$INPUT_FILE -all
echo "MYINFO: Completed freesurfer"

# collect data
cd $INITIAL_DIR/sub-101
echo "MYINFO: Starting tar"
echo tar czvf $INITIAL_DIR/$OUTPUT_FILE .
tar czvf $INITIAL_DIR/$OUTPUT_FILE .
echo "MYINFO: Completed tar"
cd $INITIAL_DIR

echo "MYINFO: Showing ls in $INITIAL_DIR"
ls -l $INITIAL_DIR
echo "MYINFO: Showing ls in $INITIAL_DIR/sub-101"
ls -l $INITIAL_DIR/sub-101

echo "MYINFO: Script is done"
# done
