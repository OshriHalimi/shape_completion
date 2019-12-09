import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from util.gen import banner


# ----------------------------------------------------------------------------------------------------------------------
#                                               	  Abstract Pytorch Wrapper
# ----------------------------------------------------------------------------------------------------------------------
class PytorchNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.use_cuda = True if (str(self.device) == "cuda" and torch.cuda.is_available()) else False

    def family_name(self):
        return self.__class__.__name__

    def forward(self, x):
        # Make it abstract
        raise NotImplementedError

    def summary(self, x_shape, batch_size=-1, print_it=True):
        # print('Summary has been called')
        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (
                    module == self) and not (hook.__code__.co_code in [f.__code__.co_code for f in module._forward_hooks.values()])):
                hooks.append(module.register_forward_hook(hook))

        if self.use_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(x_shape, tuple):
            x_shape = [x_shape]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in x_shape]
        # print(type(x[0]))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        self(*x)

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
                    if summary[layer]["trainable"] == True:
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

    def output_size(self, x_shape,cuda_allowed=True):
        t = torch.Tensor(1, *x_shape)
        if self.use_cuda and cuda_allowed:
            t = t.cuda()
        f = self.forward(torch.autograd.Variable(t))
        return int(np.prod(f.size()[1:]))

    def print_weights(self):
        banner('Weights')
        for i, weights in enumerate(list(self.parameters())):
            print(f'Layer {i} :: weight shape: {list(weights.size())}')


# ----------------------------------------------------------------------------------------------------------------------
#                                               	  Extras
# ----------------------------------------------------------------------------------------------------------------------
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)

def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def random_sample(diversity=0.5):
    def random_samp(preds, argmax=True):
        """Helper function to sample an index from a probability array"""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity

        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        # Randomly draw smaples from the multinomial distrubtion with 101 probability classes
        if argmax:
            return np.argmax(probas)
        else:
            return probas

    return random_samp


def max_sample():
    def m(preds):
        return np.argmax(preds)

    return m
