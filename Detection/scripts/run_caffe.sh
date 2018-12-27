# for model_name in VGG MobileNet MobileNetV2; do
for model_name in VGG MobileNet MobileNetV2; do
	for batch in 4; do
		for fp in FP11 FP16; do
			for precision in float; do
				./convert_model_caffe.sh $model_name $batch $precision # | grep -e '***' -e batch
				./validate_caffe.sh $fp $model_name # | grep -e Args -e Average\ infer\ time -e Mean\ Average\ Precision
				printf $"**********\n\n"
			done
		done
	done
done
