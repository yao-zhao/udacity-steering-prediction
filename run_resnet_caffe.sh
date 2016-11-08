echo "start  training-------------------------------------------------------------------------"
~/caffe3/build/tools/caffe train --solver=caffe_model/solver.prototxt \
   --weights=resnet_data/ResNet-50-model.caffemodel -gpu 1,0 2>&1 | tee log_res.txt 
