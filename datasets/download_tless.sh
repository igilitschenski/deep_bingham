#!/bin/bash

# Find directory of this script resolving symlinks as suggested in
# https://stackoverflow.com/questions/59895/get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
cur_dir="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# Create T-Less folder.
cd $cur_dir
mkdir -p t-less_v2/train_kinect/
cd t-less_v2/train_kinect/

# Download the actual dataset.
for i in $(seq 1 30); do
  objid=$(printf "%02d" $i)
  wget http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_train_kinect_$objid.zip
  unzip t-less_v2_train_kinect_$objid.zip
done
