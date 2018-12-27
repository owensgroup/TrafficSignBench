# for model_name in MobileNet MobileNet32; do
for model_name in ResNet32; do
	for batch in 1; do
		for fp in FP16; do
			for precision in half float; do
				./convert_model_mxnet.sh $model_name $batch $precision # | grep -e '***' -e batch
				./validate_mxnet.sh $fp $model_name # | grep -e Args -e Average\ infer\ time -e Mean\ Average\ Precision
				# printf $"**********\n\n"
			done
		done
	done
done
