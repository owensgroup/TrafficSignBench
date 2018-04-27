from Benchmarks.KerasBench import KerasBench
from Benchmarks.MXNetBench import MXNetBench
from Benchmarks.PyTorchBench import PyTorchBench
from Benchmarks.NeonBench import NeonBench
# from Benchmarks.CNTKBench import CNTKBench

class Bench:
	def __init__(self, args):
		self.args = args
		self.bs = {
			"keras": KerasBench,
			"mxnet": MXNetBench,
			"pytorch": PyTorchBench,
			"neon": NeonBench
			# "cntk": CNTKBench
		}

		self.root, trainImages, trainLabels, self.testImages, self.testLabels, self.class_num = DLHelper.getImageSets(args.root, (args.resize_side, args.resize_side), dataset=args.dataset, preprocessing=args.preprocessing, printing=args.printing)
		self.x_train, self.x_valid, self.y_train, self.y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

	def bench(self):
		for framework in args.frameworks:
			bm = self.bs.get(framework.lower(), "%s isn't a valid framework!".format(framework))
			b = bm(self.args, self.root, self.x_train, self.x_valid, self.y_train, self.y_valid, self.testImages, self.testLabels, self.class_num)
			b.benchmark()
			# Release GPU memory
			del b
			gc.collect()