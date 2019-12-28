import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from util.gen import banner
from torchviz import make_dot
import types
from pathlib import Path


# ----------------------------------------------------------------------------------------------------------------------
#                                               	  Abstract PyTorch Wrapper
# ----------------------------------------------------------------------------------------------------------------------
class PytorchNet(nn.Module):
    # self._use_cuda = True if (str(self._device) == "cuda" and torch.cuda.is_available()) else False
    @classmethod
    def monkeypatch(cls, o):
        # There should be some nicer method to do this - but I rather save the time instead
        o.on_gpu = types.MethodType(cls.on_gpu, o)
        o.family_name = types.MethodType(cls.family_name, o)
        o.visualize = types.MethodType(cls.visualize, o)
        o.output_size = types.MethodType(cls.output_size, o)
        o.print_weights = types.MethodType(cls.print_weights, o)
        o.summary = types.MethodType(cls.summary, o)
        o._random_input = types.MethodType(cls._random_input, o)
        o._predict_input_size = types.MethodType(cls._predict_input_size, o)

        return o

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def family_name(self):
        # For Families of Models - such as ResNet190, ResNet180 etc - FamilyName == ResNet
        return self.__class__.__name__

    def visualize(self, x_shape=None, frmt='pdf'):
        # Possible Formats: https://www.graphviz.org/doc/info/output.html
        x = self._random_input(x_shape)
        y = self.forward(*x)
        g = make_dot(y, params=dict(self.named_parameters()))
        g.format = frmt
        fp = g.view(filename=self.family_name(), cleanup=True)
        print(f'Outputted {g.format} visualization file to {Path(fp).resolve()}')

    def output_size(self, x_shape=None):
        x = self._random_input(x_shape)
        y = self.forward(*x)
        out = list(y.size()[1:])
        if len(out) == 1:
            out = out[0]
        return out

    def print_weights(self):
        banner('Weights')
        for i, weights in enumerate(list(self.parameters())):
            print(f'Layer {i} :: weight shape: {list(weights.size())}')

    def summary(self, x_shape=None, batch_size=-1, print_it=True):
        def register_hook(mod):

            # noinspection PyArgumentList
            def hook(module, xx, yy):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(xx[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(yy, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in yy
                    ]
                else:
                    summary[m_key]["output_shape"] = list(yy.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (not isinstance(mod, nn.Sequential) and not isinstance(mod, nn.ModuleList) and not (
                    mod == self) and not (
                    hook.__code__.co_code in [f.__code__.co_code for f in mod._forward_hooks.values()])):
                hooks.append(mod.register_forward_hook(hook))

        if x_shape is None:
            x_shape = self._predict_input_size()
        x = self._random_input(x_shape)

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        self.forward(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        if print_it:
            print("----------------------------------------------------------------")
            line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
            print(line_new)
            print("================================================================")
            total_params = 0
            total_output = 0
            trainable_params = 0
            for layer in summary:
                # input_shape, output_shape, trainable, nb_params
                line_new = "{:>20}  {:>25} {:>15}".format(
                    layer,
                    str(summary[layer]["output_shape"]),
                    "{0:,}".format(summary[layer]["nb_params"]),
                )
                total_params += summary[layer]["nb_params"]
                total_output += np.prod(summary[layer]["output_shape"])
                if "trainable" in summary[layer]:
                    if summary[layer]["trainable"]:
                        trainable_params += summary[layer]["nb_params"]
                print(line_new)

            # assume 4 bytes/number (float on cuda).
            total_input_size = abs(np.prod(x_shape) * batch_size * 4. / (1024 ** 2.))
            total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
            total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
            total_size = total_params_size + total_output_size + total_input_size

            print("================================================================")
            print("Total params: {0:,}".format(total_params))
            print("Trainable params: {0:,}".format(trainable_params))
            print("Non-trainable params: {0:,}".format(total_params - trainable_params))
            print("----------------------------------------------------------------")
            print("Input size (MB): %0.2f" % total_input_size)
            print("Forward/backward pass size (MB): %0.2f" % total_output_size)
            print("Params size (MB): %0.2f" % total_params_size)
            print("Estimated Total Size (MB): %0.2f" % total_size)
            print("----------------------------------------------------------------")
        return summary

    def _random_input(self, x_shape):
        dtype = torch.cuda.FloatTensor if self.on_gpu() else torch.FloatTensor

        if x_shape is None:
            x_shape = self._predict_input_size()

        # multiple inputs to the network
        if isinstance(x_shape, tuple):
            x_shape = [x_shape]

        # batch_size of 2 for batchnorm
        return [torch.rand(2, *in_size).type(dtype) for in_size in x_shape]

    def _predict_input_size(self):
        # TODO - this kinda sucks. It should depend on the type of input layer found
        print('Attempt was made to discern needed input size')
        return tuple(next(self.parameters()).size()[1:])
