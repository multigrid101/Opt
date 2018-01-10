#!/bin/bash

git checkout HEAD~1

example_dir=`pwd`
cd /home/sebastian/Optlang/Opt/API
make
cd $example_dir
make

