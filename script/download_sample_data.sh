#!/usr/bin/env bash
set -e
set -x

data_dir=data
mkdir -p $data_dir
cd $data_dir

if test -f "samples.tar" ; then
    echo "Tar file exists: samples.tar"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/rollingdepth/data/samples.tar

tar -xf samples.tar
rm samples.tar