#!/bin/bash
for d in 0.2 0.15 0.10 0.05 0.01
do
  echo $d
  ipython DecomposingTestOfMixtureModelsClassifier.py mlp $d
  cd scripts
  ipython ploting_sesion.py
  cd ../plots
  mkdir mlp/$d
  mv *.png mlp/$d
  cp mlp/*model.png mlp/$d
  cp mlp/full_signal.png mlp/$d
  cd ..
done
