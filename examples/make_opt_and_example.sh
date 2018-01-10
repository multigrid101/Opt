#!/bin/bash

# re-compiles Opt (the language) AND the current example

example_dir=`pwd`
cd /home/sebastian/Optlang/Opt/API
make clean
make -j4
cd $example_dir
make -j4

