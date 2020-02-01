import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from util.string import banner, warn
from util.fs import convert_bytes
from util.func import list_class_declared_methods
import types
from pathlib import Path
from pytorch_lightning import LightningModule
from collections.abc import Sequence
from inspect import signature
import random


# ----------------------------------------------------------------------------------------------------------------------
#                                               	  Abstract PyTorch Wrapper
# ----------------------------------------------------------------------------------------------------------------------
class PytorchNet(LightningModule):
    # self._use_cuda = True if (str(self._device) == "cuda" and torch.cuda.is_available()) else False
    @classmethod
    def monkeypatch(cls, o, force=False):
        # Check we are not overriding something:
        to_override = list_class_declared_methods(cls) - {'monkeypatch'}  # Remove the class methods
        existing_attributes = set(dir(o))  # Checks for variables as well
        intersect = existing_attributes & to_override
        if intersect:
            if force:
                warn(f"Found method collision - removing override of {intersect}")
                to_override -= intersect
            else:
                assert not intersect, f"Found Monkeypatch intersection {intersect}"

        # Do a bounding to the object o
        for meth_name in to_override:
            setattr(o, meth_name, types.MethodType(getattr(cls, meth_name), o))

        return o

    @staticmethod
    def print_memory_usage(device=0):
        print(f'Memory Usage for GPU: {torch.cuda.get_device_name(device)}')
        print('Allocated:', round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(device) / 1024 ** 3, 1), 'GB')

    @staticmethod
    def learning_rate(opt):
        # Note - this only suits a model-uniform learning rate
        # See: https://discuss.pytorch.org/t/print-current-learning-rate-of-the-adam-optimizer/15204/9
        return opt.param_groups[0]['lr']

    @staticmethod
    def identify_system():
        from platform import python_version
        from util.cpuinfo import cpu
        import psutil
        print(f'Python {python_version()} , Pytorch {torch.__version__}, CuDNN {torch.backends.cudnn.version()}')
        cpu_dict = cpu.info[0]
        mem = psutil.virtual_memory().total
        num_cores_str = f" :: {psutil.cpu_count() / psutil.cpu_count(logical=False)} Cores"
        mem_str = f" :: {convert_bytes(mem)} Memory"

        if 'ProcessorNameString' in cpu_dict:  # Windows
            cpu_name = cpu_dict['ProcessorNameString'] + num_cores_str + mem_str
        elif 'model name' in cpu_dict:  # Linux
            cpu_name = cpu_dict['model name'] + num_cores_str + mem_str
        else:
            raise NotImplementedError

        print(f'CPU : {cpu_name}')
        gpu_count = torch.cuda.device_count()
        print(f'Found {gpu_count} GPU Devices:')
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f'\tGPU {i}: {p.name} [{p.multi_processor_count} SMPs , {convert_bytes(p.total_memory)} Memory]')

    def ongpu(self):
        # Due to the lightning model on_gpu variable
        indicator = getattr(self, 'on_gpu', None)
        if indicator is None:
            return next(self.parameters()).is_cuda
        else:
            return indicator
            # This is the code:
            # self.on_gpu = True if (gpus and torch.cuda.is_available()) else False
            # Not exactly the same, in truth. It depends on the time we call ongpu()

    def family_name(self):
        # For Families of Models - such as ResNet190, ResNet180 etc - FamilyName == ResNet
        return self.__class__.__name__

    def visualize(self, x_shape=None, frmt='pdf'):
        from torchviz import make_dot
        # Possible Formats: https://www.graphviz.org/doc/info/output.html
        x = self._random_input(x_shape)
        y = self.forward(*x)
        g = make_dot(y, params=None)
        g.format = frmt
        fp = g.view(filename=self.family_name(), cleanup=True)
        print(f'Outputted {g.format} visualization file to {Path(fp).resolve()}')

    def output_size(self, x_shape=None):
        x = self._random_input(x_shape)
        y = self.forward(*x)
        out = tuple(y.size()[1:])
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
        dtype = torch.cuda.FloatTensor if self.ongpu() else torch.FloatTensor

        if x_shape is None:
            x_shape = self._predict_input_size()

        # Multiple
        if not isinstance(x_shape[0], Sequence):
            x_shape = [x_shape]

        # batch_size of 2 for batchnorm
        return [torch.rand(2, *in_size).type(dtype) for in_size in x_shape]

    def _predict_input_size(self):
        # TODO - Simply doesn't really work. Not sure it is possible to predict
        num_input_params_to_forward = len(signature(getattr(self, 'forward')).parameters)
        first_layer_size = (tuple(next(self.parameters()).size()[1:][::-1]),)
        in_shape = first_layer_size * num_input_params_to_forward
        warn(f'Experimental input size prediction : {in_shape}')
        return in_shape


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Standalone Functions
# ----------------------------------------------------------------------------------------------------------------------

def set_determinsitic_run(seed=None):
    if seed is None:
        # Specific to the ShapeCompletion platform
        from cfg import UNIVERSAL_RAND_SEED
        seed = UNIVERSAL_RAND_SEED

    # CPU Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU Seeds
    torch.cuda.manual_seed_all(seed)
    # CUDNN Framework
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Might be best to turn off benchmark for deterministic results:
    # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054


def worker_init_closure(seed=None):
    if seed is None:
        # Specific to the ShapeCompletion platform
        from cfg import UNIVERSAL_RAND_SEED
        seed = UNIVERSAL_RAND_SEED

    def worker_init_fn(worker_id):
        random.seed(worker_id + seed)
        np.random.seed(worker_id + seed)
        torch.manual_seed(seed)
        # See if this needs more work

    return worker_init_fn


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    import torchvision

    model = torchvision.models.resnet50(False)
    pymodel = PytorchNet.monkeypatch(model)
