GPU=0
REPEAT=1
while [[ $# -gt 1 ]]
do
key="$1"
case $key in
-g|--gpu)
GPU="$2"
shift # past argument
;;
-r|--repeat)
REPEAT="$2"
shift # past argument
;;
*)
# unknown option
;;
esac
shift # past argument
done
/home/yz/caffe3/build/tools/caffe train -gpu $GPU \
--solver=models/net5/solver_0.prototxt \
2>&1 | tee models/net5/log_$REPEAT.txt
cp models/net5/stage_0_iter_80000.caffemodel \
models/net5/stage_0_final_$REPEAT.caffemodel
