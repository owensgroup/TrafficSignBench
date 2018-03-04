import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--root", help="Root directory of the dataset", 
		type=str, required=True)
	parser.add_argument("--network_type", help="Type of neural network model", 
		type=str, default="idsia")
	parser.add_argument("--resize_side", help="Size of the resized image", 
		type=int, default=48)
	parser.add_argument("--dataset", help="Dataset", 
		type=str, default="GT")
	parser.add_argument("--epoch_num", help="Size of the resized image", 
		type=int, default=25)
	parser.add_argument("--batch_size", help="Batch size", 
		type=int, default=64)
	parser.add_argument("--preprocessing", help="Preprocessing type", 
		type=str)
	parser.add_argument("--printing", help="If print the result", 
		type=bool, default=False)
	parser.add_argument("-d", "--devices", action="append", help="Device (CPU/GPU)", 
		type=str, required=True)
	parser.add_argument("-b", "--backends", action="append", help="Backends", 
		type=str)
	parser.add_argument("-f", "--frameworks", action="append", help="Frameworks to be benchmarked", 
		type=str, required=True)

	args = parser.parse_args()
	benchObj = Bench(args)
	benchObj.bench()