#!/bin/bash

# re-compiles Opt (the language) and ALL examples, including a 'make clean'

homedir=/home/sebastian/Optlang/Opt/examples

cd /home/sebastian/Optlang/Opt/API
make clean
make
cd $homedir

for example_dir in arap_mesh_deformation cotangent_mesh_smoothing embedded_mesh_deformation image_warping intrinsic_image_decomposition optical_flow poisson_image_editing robust_nonrigid_alignment shape_from_shading volumetric_mesh_deformation
do

  # echo "changing to $example_dir"
  cd $example_dir
  # make clean
  make
  cd $homedir

done
