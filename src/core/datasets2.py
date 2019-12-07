import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
from gen_utils import banner

# ----------------------------------------------------------------------------------------------------------------------
#                                                Base Class
# ----------------------------------------------------------------------------------------------------------------------
class ClassificationDataset:
    def __init__(self, data_dir, class_labels, shape, testset_size, trainset_size, dataset_space, expected_files):
        # Basic Dataset Info
        self._class_labels = tuple(class_labels)
        self._shape = tuple(shape)
        self._testset_size = testset_size
        self._trainset_size = trainset_size
        self._dataset_space = dataset_space
        self._data_dir = data_dir

        if not isinstance(expected_files, list):
            self._expected_files = [expected_files]
        else:
            self._expected_files = expected_files

        self._download = True if any(
            not os.path.isfile(os.path.join(self._data_dir, file)) for file in self._expected_files) else False

    def data_summary(self, show_sample=False):
        img_type = 'Grayscale' if self._shape[0] == 1 else 'Color'
        banner('Dataset Summary')
        print(f'\n* Dataset Name: {self.name()} , {img_type} images')
        print(f'* Data shape: {self._shape}')
        print(f'* Training Set Size: {self._trainset_size} samples')
        print(f'* Test Set Size: {self._testset_size} samples')
        print(f'* Estimated Hard-disk space required: ~{convert_bytes(self._dataset_space)}')
        print(f'* Number of classes: {self.num_classes()}')
        print(f'* Class Labels:\n{self._class_labels}')
        banner()
        if show_sample:
            self.trainset(show_sample=True)

    def name(self):
        assert self.__class__.__name__ != 'ClassificationDataset'
        return self.__class__.__name__

    def dataset_space(self):
        return self._dataset_space

    def num_classes(self):
        return len(self._class_labels)

    def class_labels(self):
        return self._class_labels

    def input_channels(self):
        return self._shape[0]

    def shape(self):
        return self._shape

    def max_test_size(self):
        return self._testset_size

    def max_train_size(self):
        return self._trainset_size

    def testset(self, batch_size, max_samples=None, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 1, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4, False

        test_dataset = self._test_importer()
        # print(sum(1 for _ in test_dataset))

        if max_samples < self._testset_size:
            testset_siz = max_samples
            test_sampler = SequentialSampler(list(range(max_samples)))
        else:
            test_sampler = None
            testset_siz = self._testset_size

        test_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

        return test_gen, testset_siz

    def trainset(self, batch_size=128, valid_size=0.1, max_samples=None, augment=True, shuffle=True, random_seed=None,
                 show_sample=False, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 1, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4, False

        max_samples = self._trainset_size if max_samples is None else min(self._trainset_size, max_samples)
        assert ((valid_size >= 0) and (valid_size <= 1)), "[!] Valid_size should be in the range [0, 1]."

        train_dataset = self._train_importer(augment)
        # print(sum(1 for _ in train_dataset)) #Can be used to discover the trainset size if needed

        val_dataset = self._train_importer(False)  # Don't augment validation

        indices = list(range(self._trainset_size))
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)

        indices = indices[:max_samples]  # Truncate to desired size
        # Split validation
        split = int(np.floor(valid_size * max_samples))
        train_ids, valid_ids = indices[split:], indices[:split]

        num_train = len(train_ids)
        num_valid = len(valid_ids)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)

        if show_sample: self._show_sample(train_dataset, 4)

        return (train_loader, num_train), (valid_loader, num_valid)

    def _show_sample(self, train_dataset, siz):
        images, labels = iter(torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=siz ** 2)).next()
        plot_images(images.numpy().transpose([0, 2, 3, 1]), labels, self._class_labels, siz=siz)

    def _train_importer(self, augment):
        raise NotImplementedError

    def _test_importer(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Implementations
# ----------------------------------------------------------------------------------------------------------------------

class CIFAR10(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(
            data_dir=data_dir,
            class_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                          'truck'],
            shape=(3, 32, 32),
            testset_size=10000,
            trainset_size=50000,
            dataset_space=170500096,
            expected_files=os.path.join('CIFAR10', 'cifar-10-python.tar.gz')
        )

    def _train_importer(self, augment):
        ops = [transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
        if augment:
            ops.insert(0, transforms.RandomCrop(32, padding=4))
            ops.insert(0, transforms.RandomHorizontalFlip())
        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=True, download=self._download,
                                transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=False, download=self._download,
                                transform=transforms.Compose(ops))


class MNIST(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                         shape=(1, 28, 28), testset_size=10000, trainset_size=60000, dataset_space=110403584,
                         expected_files=[os.path.join('MNIST', 'processed', 'training.pt'),
                                         os.path.join('MNIST', 'processed', 'test.pt')])

    def _train_importer(self, augment):  # Convert 1 channels -> 3 channels #transforms.Grayscale(3),
        ops = [transforms.ToTensor(),
               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        return datasets.MNIST(root=os.path.join(self._data_dir, 'MNIST'), train=True, download=self._download,
                              transform=transforms.Compose(ops))

    def _test_importer(self):  # Convert 1 channels -> 3 channels
        ops = [transforms.ToTensor(),
               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        return datasets.MNIST(root=os.path.join(self._data_dir, 'MNIST'), train=False, download=self._download,
                              transform=transforms.Compose(ops))


class FashionMNIST(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                       'Sneaker', 'Bag', 'Ankle boot'],
                         shape=(1, 28, 28), testset_size=10000, trainset_size=60000, dataset_space=110403584,
                         expected_files=[os.path.join('FashionMNIST', 'processed', 'training.pt'),
                                         os.path.join('FashionMNIST', 'processed', 'test.pt')])

    def _train_importer(self, augment):  # Convert 1 channels -> 3 channels #transforms.Grayscale(3),
        ops = [transforms.ToTensor()]
        return datasets.FashionMNIST(root=os.path.join(self._data_dir, 'FashionMNIST'), train=True,
                                     download=self._download,
                                     transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.ToTensor()]
        return datasets.FashionMNIST(root=os.path.join(self._data_dir, 'FashionMNIST'), train=False,
                                     download=self._download,
                                     transform=transforms.Compose(ops))


class STL10(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship',
                                       'truck'],
                         shape=(3, 96, 96), testset_size=8000, trainset_size=5000, dataset_space=2640400384,
                         expected_files=os.path.join('STL10', 'stl10_binary.tar.gz'))

    def _train_importer(self, augment):
        ops = [transforms.ToTensor()]
        return datasets.STL10(root=os.path.join(self._data_dir, 'STL10'), split='train', download=self._download,
                              transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.ToTensor()]
        return datasets.STL10(root=os.path.join(self._data_dir, 'STL10'), split='test', download=self._download,
                              transform=transforms.Compose(ops))


class TinyImageNet(ClassificationDataset):  # TODO - Implement support for Val Directory
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=TINY_IMAGENET_NAMES,
                         shape=(3, 64, 64), testset_size=10000, trainset_size=100000, dataset_space=497799168,
                         expected_files=[os.path.join('TinyImageNet', 'train'), os.path.join('TinyImageNet', 'test')])

    def _train_importer(self, augment):
        ops = [transforms.ToTensor()]
        if augment:
            ops.insert(0, transforms.RandomHorizontalFlip())

        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'TinyImageNet', 'train'),
                                    transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.ToTensor()]
        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'TinyImageNet', 'test'),
                                    transform=transforms.Compose(ops))


class ImageNet(ClassificationDataset):
    # TODO- There might be a mismatch between class labels and the IMAGE_NETLABEL_NAMES - See TinyImageNet_label_names()
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=IMAGE_NETLABEL_NAMES,
                         shape=(3, 256, 256), testset_size=10000, trainset_size=100000, dataset_space=497799168,
                         expected_files=[os.path.join('ImageNet', 'train'), os.path.join('ImageNet', 'test')])

    def _train_importer(self, augment):
        ops = [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        if augment:
            ops.insert(0, transforms.Resize(256))
            ops.insert(0, transforms.RandomResizedCrop(224))
            ops.insert(0, transforms.RandomHorizontalFlip())
        else:
            ops.insert(0, transforms.Resize(256))
            ops.insert(0, transforms.CenterCrop(224))

        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'ImageNet', 'train'),
                                    transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
               transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'ImageNet', 'test'),
                                    transform=transforms.Compose(ops))


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Implementations
# ----------------------------------------------------------------------------------------------------------------------
class Datasets:
    _implemented = {
        'MNIST': MNIST,
        'CIFAR10': CIFAR10,
        'ImageNet': ImageNet,
        'TinyImageNet': TinyImageNet,
        'STL10': STL10,
        'FashionMNIST': FashionMNIST
    }

    @staticmethod
    def which():
        return tuple(Datasets._implemented.keys())

    @staticmethod
    def get(dataset_name, data_dir):
        return Datasets._implemented[dataset_name](data_dir)


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def plot_images(images, cls_true, label_names, cls_pred=None, siz=3):
    # Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    fig, axes = plt.subplots(siz, siz)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :].squeeze(), interpolation='spline16', cmap='gray')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = f"{cls_true_name} ({cls_true[i]})"
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = f"True: {cls_true_name}\nPred: {cls_pred_name}"
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()



class DatasetMenu:
    _implemented = {
        # 'MNIST': MNIST,
        # 'CIFAR10': CIFAR10,
        # 'ImageNet': ImageNet,
        # 'TinyImageNet': TinyImageNet,
        # 'STL10': STL10,
        # 'FashionMNIST': FashionMNIST
    }

    @staticmethod
    def which():
        return tuple(DatasetMenu._implemented.keys())

    @staticmethod
    def get(dataset_name, data_dir):
        return DatasetMenu._implemented[dataset_name](data_dir)
# ----------------------------------------------------------------------------------------------------------------------
#                                                Base Class
# ----------------------------------------------------------------------------------------------------------------------
class ProjectionDataset:
    def __init__(self, projection_dir,mesh_dir,subject_ids,num_proj_per_mesh,):
        # Basic Dataset Info
        mesh_dir = Path(mesh_dir)
        projection_dir = Path(projection_dir)
        assert_is_dir(mesh_dir)
        assert_is_dir(projection_dir)

        self._mesh_dir =projection_dir
        self._projection_dir = projection_dir
        self._num_proj_per_mesh = num_proj_per_mesh
        self._subject_ids = subject_ids

    def data_summary(self):
        banner('Dataset Summary')
        print(f'\n* Dataset Name: {self.name()}')
        print(f'* Dataset size: {self._num_projections} Projections')
        print(f'* Test Set Size: {self._testset_size} samples')
        print(f'* Number of classes: {self.num_classes()}')
        print(f'* Class Labels:\n{self._class_labels}')
        banner()


    def name(self):
        assert self.__class__.__name__ != 'PointDataset'
        return self.__class__.__name__

    def max_test_size(self):
        return self._testset_size

    def max_train_size(self):
        return self._trainset_size

    def testset(self, batch_size, max_samples=None, device='cuda'):

        num_workers = 4
        if device.lower() == 'cuda' and torch.cuda.is_available():
            pin_memory = True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            pin_memory = False

        test_dataset = self._test_importer()
        # print(sum(1 for _ in test_dataset))

        if max_samples < self._testset_size:
            testset_siz = max_samples
            test_sampler = SequentialSampler(list(range(max_samples)))
        else:
            test_sampler = None
            testset_siz = self._testset_size

        test_gen = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

        return test_gen, testset_siz

    def trainset(self, batch_size=128, valid_size=0.1, max_samples=None, augment=True, shuffle=True, random_seed=None,
                 show_sample=False, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 1, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4, False

        max_samples = self._trainset_size if max_samples is None else min(self._trainset_size, max_samples)
        assert ((valid_size >= 0) and (valid_size <= 1)), "[!] Valid_size should be in the range [0, 1]."

        train_dataset = self._train_importer(augment)
        # print(sum(1 for _ in train_dataset)) #Can be used to discover the trainset size if needed

        val_dataset = self._train_importer(False)  # Don't augment validation

        indices = list(range(self._trainset_size))
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)

        indices = indices[:max_samples]  # Truncate to desired size
        # Split validation
        split = int(np.floor(valid_size * max_samples))
        train_ids, valid_ids = indices[split:], indices[:split]

        num_train = len(train_ids)
        num_valid = len(valid_ids)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)

        if show_sample: self._show_sample(train_dataset, 4)

        return (train_loader, num_train), (valid_loader, num_valid)

    def _show_sample(self, train_dataset, siz):
        images, labels = iter(torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=siz ** 2)).next()
        plot_images(images.numpy().transpose([0, 2, 3, 1]), labels, self._class_labels, siz=siz)

    def _train_importer(self, augment):
        raise NotImplementedError

    def _test_importer(self):
        raise NotImplementedError
