import torch
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

class FiveCrop(object):
    """Class for getting five crops from PIL.Image data."""
    def __init__(self, size):
        """
        Args:
            size (Union[integer, tuple]): Size of the five crops, resulting in a (W, H) size.
        """
        if type(size) == int:
            size = (size, size)
        else:
            assert len(size) == 2
        self.size = size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image): Image to be cropped.

        Returns:
            tuple: Contains PIL.Image for the five crops.
        """
        # For PIL Image format
        w, h = image.size
        crop_w, crop_h = self.size
        if crop_w > w or crop_h > h:
            raise ValueError("Crop size is bigger than image size.")

        image = np.asarray(image, np.uint8)

        top_left = Image.fromarray(image[:crop_h,:crop_w,:], mode='RGB')
        top_right = Image.fromarray(image[:crop_h,w-crop_w:,:], mode='RGB')
        bottom_left = Image.fromarray(image[h-crop_h:,:crop_w,:], mode='RGB')
        bottom_right = Image.fromarray(image[h-crop_h:,w-crop_w:,:], mode='RGB')

        left = int(np.ceil((w - crop_w)/2))
        top = int(np.ceil((h - crop_h)/2))
        right = int(np.ceil((w + crop_w)/2))
        bottom = int(np.ceil((h + crop_h)/2))
        center = Image.fromarray(image[top:bottom,left:right,:], mode='RGB')

        return (top_left, top_right, bottom_left, bottom_right, center)


class ToTensor(object):
    """Class for converting PIL.Image data to Tensor."""
    def __call__(self, image):
        """
        Args:
            image (PIL.Image): Image to be converted.

        Returns:
            Tensor: Tensor image of size (C, H, W).
        """
        image = np.asarray(image, np.uint8)
        image = torch.from_numpy(image)

        # Convert from HWC to CHW format
        image = image.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image


class Normalize(object):
    """Class for normalizing image values to have a mean of zero and a variance of one."""
    def __init__(self, mean, std):
        """
        Args:
            mean (list): Mean values for image channels.
            std (list): Standard deviation values for image channels.
        """
        assert len(mean) >= 1
        assert len(std) >= 1

        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if len(self.mean) == 1:
            for i in range(tensor.shape[0]):
                tensor[i] -= self.mean[0]
        elif len(self.mean) != 1:
            assert len(self.mean) == tensor.shape[0]
            for i in range(tensor.shape[0]):
                tensor[i] -= self.mean[i]

        if len(self.std) == 1:
            for i in range(tensor.shape[0]):
                tensor[i] /= self.std[0]
        elif len(self.std) != 1:
            assert len(self.std) == tensor.shape[0]
            for i in range(tensor.shape[0]):
                tensor[i] /= self.std[i]

        return tensor

class ImageNetDataset(Dataset):
    """Dataloader class for the first 2500 ImageNet images."""
    def __init__(self, root_dir, data_limit, smaller_side_size, normalize=True, crop_size=224, crop_type='center'):
        """
        Args:
            root_dir (string): 'imagenet_first2500' directory with all compressed data extracted.
            data_limit (integer): Maximum number of images to get from 2500 images, in numerical order.
            smaller_side_size (integer): Size of smaller side of image for resizing.
            normalize (boolean): Whether to normalize image values to have a mean of zero and a variance of one.
            crop_size (integer): Size of image crop(s).
            crop_type (string): Whether to do a center crop, a five crop, or a self-implemented FiveCrop + ToTensor + Normalize.
        """
        assert type(data_limit) == int

        self.root_dir = os.path.join(root_dir, '')

        label_folder = self.root_dir + 'ILSVRC2012_bbox_val_v3/val/'
        synset_words = self.root_dir + 'synset_words.txt'
        self.image_dir = os.path.join(self.root_dir, 'imagenet2500/imagespart')

        self.x, self.y = self.create_dataset(synset_words, label_folder, data_limit)

        self.transform = self.create_transform(smaller_side_size=smaller_side_size, normalize=normalize, crop_size=crop_size, crop_type=crop_type)

    def __len__(self):
        """
        Returns:
            integer: Total number of samples.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Args:
            idx (integer): Index of sample to be predicted.

        Returns:
            dictionary: Tensor and label of one sample.
        """
        img_name = os.path.join(self.image_dir, self.x[idx])
        image = Image.open(img_name) # (channels, width, height)
        if image.mode == 'L':
            image = image.convert(mode='RGB')
        image = self.transform(image)
        label = self.y[idx]
        sample = {'image': image, 'label': label}

        return sample

    def create_dataset(self, synset_words, label_folder, data_limit=2500):
        """
        Args:
            synset_words (string): Contains class numbers.
            label_folder (string): Folder containing labels of all 50000 ImageNet samples.

        Returns:
            list: Names of all considered image samples.
            list: Labels of all considered image samples.
        """
        assert type(data_limit) == int
        assert data_limit <= 2500 and data_limit > 0

        # synsets_to_class_descriptions={}
        # idx_to_synsets = {}
        x = []
        y = []

        synsets_to_idx = {}
        ct=-1
        with open(synset_words) as f:
          for line in f:
            if (len(line)> 5):
              z=line.strip().split()
              descr=''
              for i in range(1,len(z)):
                descr=descr+' '+z[i]
              
              ct+=1
              # idx_to_synsets[ct]=z[0]
              synsets_to_idx[z[0]]=ct
              # synsets_to_class_descriptions[z[0]]=descr[1:]

        img_names = []
        for i in range(data_limit):
            img_names.append('ILSVRC2012_val_' + str(i+1).zfill(8))

        for img_name in img_names:
            tree = ET.parse(os.path.join(label_folder, img_name + '.xml'))
            root = tree.getroot()

            lbset=set()
            
            for obj in root.findall('object'):
              for name in obj.findall('name'):
                ind = synsets_to_idx[name.text]
                # first_name = name.text
                lbset.add(ind)
                
            if len(lbset)!=1:
              print('ERR: len(lbset)!=1',  len(lbset))
              exit()
              
            for s in lbset:
              label = s

            x.append(img_name + '.JPEG')
            y.append(label)

        return x, y

    def create_transform(self, smaller_side_size, normalize, crop_size, crop_type):
        """
        Args:
            smaller_side_size (integer): Size of smaller side of image for resizing.
            normalize (boolean): Whether to normalize image values to have a mean of zero and a variance of one.
            crop_size (integer): Size of image crop(s).
            crop_type (string): Whether to do a center crop, a five crop, or a self-implemented FiveCrop + ToTensor + Normalize.

        Returns:
            torchvision.transforms.Compose: Transformations to be made on image samples.
        """
        # Create transformations to rescale and crop images
        assert type(crop_size) == int

        transforms_list = []

        transforms_list.append(transforms.Resize(smaller_side_size))

        normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if crop_type == 'center':
            transforms_list.append(transforms.CenterCrop(crop_size))
            transforms_list.append(transforms.ToTensor())
            if normalize:
                transforms_list.append(normalizer)
        elif crop_type == 'five':
            transforms_list.append(transforms.FiveCrop(crop_size))
            if normalize:
                transforms_list.append(transforms.Lambda(lambda crops: torch.stack([normalizer(transforms.ToTensor()(crop)) for crop in crops])))
            else:
                transforms_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        elif crop_type == 'bonus':
            normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            transforms_list.append(FiveCrop(crop_size))
            if normalize:
                transforms_list.append(transforms.Lambda(lambda crops: torch.stack([normalizer(ToTensor()(crop)) for crop in crops])))
            else:
                transforms_list.append(transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])))

        return transforms.Compose(transforms_list)


class Model():
    """Model class for ImageNet prediction."""
    def __init__(self, model):
        """
        Args:
            model (torchvision.models.*): Pretrained model from PyTorch's model zoo.
        """
        self.model = model
        self.arch = None

    def predict(self, testloader, data_limit, smaller_side_size, normalize=True, crop_size=224, crop_type='center'):
        """
        Args:
            testloader (torch.utils.data.DataLoader): PyTorch dataloader for model prediction.
            data_limit (integer): Maximum number of images to get from 2500 images, in numerical order.
            smaller_side_size (integer): Size of smaller side of image for resizing.
            normalize (boolean): Whether to normalize image values to have a mean of zero and a variance of one.
            crop_size (integer): Size of image crop(s).
            crop_type (string): Whether to do a center crop, a five crop, or a self-implemented FiveCrop + ToTensor + Normalize.
        """
        self.model.eval()

        if self.arch is not None:
            print("Doing prediction with " + str(self.arch) + " for " + str(data_limit) + " samples resized to " + str(smaller_side_size) + " with normalization=" + str(normalize) + " and " + str(crop_type) + " crop of size " + str(crop_size) + "...")
        else:
            print("Doing prediction for " + str(data_limit) + " samples resized to " + str(smaller_side_size) + " with normalization=" + str(normalize) + " and " + str(crop_type) + " crop of size " + str(crop_size) + "...")

        correct = 0
        total = 0
        
        # Calculate prediction accuracies
        with torch.no_grad():
            for data in testloader:
                images, labels = data.values()
                if crop_type == 'center':
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                elif crop_type == 'five' or crop_type == 'bonus':
                    for i in range(labels.size(0)):
                        outputs = self.model(images[i])
                        avg_outputs = torch.mean(outputs, 0)
                        avg_outputs = torch.unsqueeze(avg_outputs, 0)
                        _, predicted = torch.max(avg_outputs.data, 1)
                        correct += (predicted == labels[i]).sum().item()
                    total += labels.size(0)

        print('Accuracy: ' + str(100 * correct / total) + '\n')


def run_experiment(imagenet_first2500_path, model, params, batch_size, shuffle, num_workers):
    """
    Args:
        imagenet_first2500_path (string): 'imagenet_first2500' directory with all compressed data extracted.
        model (torchvision.models.*): Pretrained model from PyTorch's model zoo.
        params (list): List of parameters for ImageNetDataset class.
        batch_size (integer): How many samples per batch to load.
        shuffle (boolean): Set to True to have the data reshuffled at every epoch.
        num_workers (integer): How many subprocesses to use for data loading.
    """
    testset = ImageNetDataset(imagenet_first2500_path, *params)
    testloader = DataLoader(testset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)
    model.predict(testloader, *params)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path of your "imagenet_first2500" folder with all compressed folders extracted')
    args = parser.parse_args()
    if args.data_dir is None:
        exit('Please make sure you have entered the path of your "imagenet_first2500" folder with all compressed folders extracted for the "--data_dir" argument.')
    imagenet_first2500_path = args.data_dir

    # Experiment parameters
    batch_size = 4
    shuffle = False
    num_workers = 2

    # First pretrained model architecture (ResNet-18)
    resnet18 = Model(models.resnet18(pretrained=True))
    resnet18.arch = 'ResNet-18'
    # Second pretrained model architecture (DenseNet-121)
    densenet = Model(models.densenet121(pretrained=True))
    densenet.arch = "DenseNet-121"

    # Experiments
    experiments = [
        # params = [model_architecture, data_limit, smaller_side_size, normalize, crop_size, crop_type]
        # Problem 1
        [resnet18, 250, 224, True, 224, 'center'],
        [resnet18, 250, 224, False, 224, 'center'],
        # Problem 2
        [resnet18, 250, 256, True, 224, 'five'],
        [resnet18, 250, 256, False, 224, 'five'],
        [resnet18, 250, 280, True, 224, 'five'],
        [resnet18, 250, 280, False, 224, 'five'],
        # Problem 2 Bonus
        [resnet18, 250, 256, True, 224, 'bonus'],
        [resnet18, 250, 256, False, 224, 'bonus'],
        [resnet18, 250, 280, True, 224, 'bonus'],
        [resnet18, 250, 280, False, 224, 'bonus'],
        # Problem 3
        [resnet18, 250, 330, True, 330, 'center'],
        [resnet18, 250, 330, False, 330, 'center'],
        [densenet, 250, 330, True, 330, 'center'],
        [densenet, 250, 330, False, 330, 'center']
    ]

    for params in experiments:
        run_experiment(imagenet_first2500_path, params[0], params[1:], batch_size, shuffle, num_workers)


if __name__ == '__main__':
    main()