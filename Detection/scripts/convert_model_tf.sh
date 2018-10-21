cd $OPENVINO_ROOTDIR/deployment_tools/model_optimizer/install_prerequisites
# ./install_prerequisites.sh tf venv
source ../venv/bin/activate

# classification
# model=idsia_48by48.pb
# model=resnet-20_32by32.pb
model=resnet-32_64by64.pb
cd ../classification/tf
# python3 ../../mo_tf.py --help --input_model $model -b 1 --input=idsia/input_placeholder --output=idsia/probs --data_type half --scale 255
# python3 ../../mo_tf.py --input_model $model -b 1 --input=resnet-20/input_placeholder --output=resnet-20/probs --data_type float --scale 255
python3 ../../mo_tf.py --input_model $model -b 32 --input=resnet-32/input_placeholder --output=resnet-32/probs --data_type float --scale 255

