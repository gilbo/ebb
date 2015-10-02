#!/usr/bin/env bash

# NOTE: To run this script, need to move it to/ make a symlink in project root
# directory, that is, liszt-ebb. Also, the models in the config files that
# this script uses are only on lightroast, so will need to update the config
# files to run this elsewhere.

# for comparison against vega (stvk), use the following meshes/ configs
# (as submitted in siggraph draft):
#   - dragon_medium_0/ dragon.config (1.3M tets)
#   - hose_medium_0  / hose.config   (1.3M tets)
#   - turtle_small_1 / turtle.config (0.9M tets)

# for comparison against freefem (neohookean), use the following meshes/ configs
# (need *.mesh medit file-format for freefem):
#   -
#   -
#   -

steps=5

outdir="ebb_times"
mkdir -p $outdir

for model in "turtle" "dragon" "hose" # stvk
# for model in "sphere" "bunny"         # neohookean
do
  config="devapps/fem/configs/${model}.config"
  for force in "stvk"
  do
    outfile="${outdir}/log_${model}_${force}_cpu"
    rm -rf $outfile
    echo "Running $config  with $force force model, $steps steps, cpu ..."
    command=`./ebb devapps/fem/sim_main.t -config  $config -force $force -steps $steps >> ${outfile}`
    $command
    outfile="${outdir}/log_${model}_${force}_gpu"
    rm -rf $outfile
    echo "Running $config  with $force force model, $steps steps, gpu ..."
    command=`./ebb --gpu devapps/fem/sim_main.t -config  $config -force $force -steps $steps >> ${outfile}`
    $command
  done
done
