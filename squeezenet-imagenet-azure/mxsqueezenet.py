# import the necessary packages
import mxnet as mx

class MxSqueezeNet:
	@staticmethod
	def squeeze(input, numFilter):
		# the first part of a FIRE module consists of a number of 1x1
		# filter squeezes on the input data followed by an activation
		squeeze_1x1 = mx.sym.Convolution(data=input, kernel=(1, 1),
			stride=(1, 1), num_filter=numFilter)
		act_1x1 = mx.sym.LeakyReLU(data=squeeze_1x1,
			act_type="elu")

		# return the activation for the squeeze
		return act_1x1

	@staticmethod
	def fire(input, numSqueezeFilter, numExpandFilter):
		# construct the 1x1 squeeze followed by the 1x1 expand
		squeeze_1x1 = MxSqueezeNet.squeeze(input, numSqueezeFilter)
		expand_1x1 = mx.sym.Convolution(data=squeeze_1x1,
			kernel=(1, 1), stride=(1, 1), num_filter=numExpandFilter)
		relu_expand_1x1 = mx.sym.LeakyReLU(data=expand_1x1,
			act_type="elu")

		# construct the 3x3 expand
		expand_3x3 = mx.sym.Convolution(data=squeeze_1x1, pad=(1, 1),
			kernel=(3, 3), stride=(1, 1), num_filter=numExpandFilter)
		relu_expand_3x3 = mx.sym.LeakyReLU(data=expand_3x3,
			act_type="elu")

		# the output of the FIRE module is the concatenation of the
		# activation for the 1x1 and 3x3 expands along the channel
		# dimension
		output = mx.sym.Concat(relu_expand_1x1, relu_expand_3x3,
			dim=1)

		# return the output of the FIRE module
		return output

	@staticmethod
	def build(classes):
		# data input
		data = mx.sym.Variable("data")

		# Block #1: CONV => RELU => POOL
		conv_1 = mx.sym.Convolution(data=data, kernel=(7, 7),
			stride=(2, 2), num_filter=96)
		relu_1 = mx.sym.LeakyReLU(data=conv_1, act_type="elu")
		pool_1 = mx.sym.Pooling(data=relu_1, kernel=(3, 3),
			stride=(2, 2), pool_type="max")

		# Block #2-4: (FIRE * 3) => POOL
		fire_2 = MxSqueezeNet.fire(pool_1, numSqueezeFilter=16,
			numExpandFilter=64)
		fire_3 = MxSqueezeNet.fire(fire_2, numSqueezeFilter=16,
			numExpandFilter=64)
		fire_4 = MxSqueezeNet.fire(fire_3, numSqueezeFilter=32,
			 numExpandFilter=128)
		pool_4 = mx.sym.Pooling(data=fire_4, kernel=(3, 3),
			stride=(2, 2), pool_type="max")

		# Block #5-8: (FIRE * 4) => POOL
		fire_5 = MxSqueezeNet.fire(pool_4, numSqueezeFilter=32,
			numExpandFilter=128)
		fire_6 = MxSqueezeNet.fire(fire_5, numSqueezeFilter=48,
			numExpandFilter=192)
		fire_7 = MxSqueezeNet.fire(fire_6, numSqueezeFilter=48,
			numExpandFilter=192)
		fire_8 = MxSqueezeNet.fire(fire_7, numSqueezeFilter=64,
			numExpandFilter=256)
		pool_8 = mx.sym.Pooling(data=fire_8, kernel=(3, 3),
			stride=(2, 2), pool_type="max")

		# Block #9-10: FIRE => DROPOUT => CONV => RELU => POOL
		fire_9 = MxSqueezeNet.fire(pool_8, numSqueezeFilter=64,
			numExpandFilter=256)
		do_9 = mx.sym.Dropout(data=fire_9, p=0.5)
		conv_10 = mx.sym.Convolution(data=do_9, num_filter=classes,
			kernel=(1, 1), stride=(1, 1))
		relu_10 = mx.sym.LeakyReLU(data=conv_10, act_type="elu")
		pool_10 = mx.sym.Pooling(data=relu_10, kernel=(13, 13),
			pool_type="avg")

		# softmax classifier
		flatten = mx.sym.Flatten(data=pool_10)
		model = mx.sym.SoftmaxOutput(data=flatten, name="softmax")

		# return the network architecture
		return model