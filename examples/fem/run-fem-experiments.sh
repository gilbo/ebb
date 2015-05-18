#!/usr/bin/env bash

# NOTE: To run this script, need to move it to/ make a symlink in project root
# directory, that is, liszt-in-terra. Also, the models in the config files that
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

outdir="vega_times"
mkdir -p $outdir

for model in "turtle" "dragon" "hose"
do
  config="examples/fem/configs/${model}.config"
  for force in "stvk"
  do
    outfile="${outdir}/log_${model}_${force}_cpu"
    echo "Running $config  with $force force model, $steps steps, cpu ..."
    command=`./liszt examples/fem/sim_main.t -config  $config -force $force -steps $steps >> ${outfile}`
    $command
    outfile="${outdir}/log_${model}_${force}_gpu"
    echo "Running $config  with $force force model, $steps steps, gpu ..."
    command=`./liszt --gpu examples/fem/sim_main.t -config  $config -force $force -steps $steps >> ${outfile}`
    $command
  done
done
