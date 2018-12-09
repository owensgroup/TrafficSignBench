for model_name in ResNet32; do
	for batch in 1; do
		for fp in FP16; do
			for precision in float; do
				./convert_model_tf.sh $model_name $batch $precision | grep -e '***' -e batch
				./validate_tf.sh $fp $model_name | grep -e Args -e Average\ infer\ time -e Top
				printf $"**********\n\n"
			done
		done
	done
done
