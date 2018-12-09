for model_name in VGG; do
	for batch in 1; do
		for fp in FP11; do
			for precision in float; do
				./convert_model_mxnet.sh $model_name $batch $precision | grep -e '***' -e batch
				./validate_mxnet.sh $fp $model_name | grep -e Args -e Average\ infer\ time -e Mean\ Average\ Precision
				printf $"**********\n\n"
			done
		done
	done
done
