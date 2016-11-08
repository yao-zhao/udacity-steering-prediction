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
set -e
/home/yz/caffe3/build/tools/caffe train -gpu $GPU \
--solver=models/net4/solver_checking_0.prototxt \
2>&1 | tee models/net4/log_checking.txt
rm models/net4/stage_0_iter_1.*
set +e
