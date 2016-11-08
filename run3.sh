
# echo "start  training-------------------------------------------------------------------------"
# ~/caffe3/build/tools/caffe train --solver=caffe_model/solver.prototxt \
#    --weights=resnet_data/ResNet-50-model.caffemodel -gpu 1,0 2>&1 | tee log.txt 

#  echo " training done "

# set caffe path
caffe_path=~/caffe3/
export PYTHONPATH="${caffe_path}python:$PYTHONPATH"
source ~/.bashrc

# set initial param
force_precheck=true
force_train=true
gpu=0
allnetnames=("net3")

python3 gen_net3.py

set -e
export GLOG_minloglevel=2
for netname in ${allnetnames[@]}; do
if [ -f "models/$netname/log_checking.txt" ] && [ ! $force_precheck == true ];
then
  echo "$netname log already exist, skip pretraining checking"
else
  echo "$netname start checking----------------------------------------------"
  bash models/$netname/checking.sh -g $gpu
  echo "$netname checking done ----------------------------------------------"
fi
done
set +e

for netname in ${allnetnames[@]}; do
  echo "$netname ------------------------------------------------------------"

export GLOG_minloglevel=0
if [ -f "models/$netname/log_$i.txt" ] && [ ! $force_train == true ];
then
  echo "$netname log already exist, skip training"
else
  echo "$netname start training----------------------------------------------"
  bash models/$netname/runfile.sh -g $gpu -r 1
  echo "$netname training done ----------------------------------------------"
fi

done