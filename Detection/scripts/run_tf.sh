# for model_name in VGG16 MobileNet MobileNetV2 ResNet18 ResNet50 SqueezeNet11; do
for model_name in VGG16 ResNet18 ResNet50 SqueezeNet11; do
	for batch in 1 4; do
# for model_name in ResNet20 ResNet32; do
# 	for batch in 1 32; do
		for fp in FP11 FP16; do
			for precision in half float; do
				./convert_model_tf.sh $model_name $batch $precision | grep -e '***' -e batch
				./validate_tf.sh $fp $model_name | grep -e Args -e Average\ infer\ time -e Mean\ Average\ Precision
				printf $"**********\n\n"
			done
		done
	done
done
