import h5py
import numpy as np
from torch.utils.data import Dataset

class KiTS(Dataset):
    """ KiTS Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if split == 'train':
            with open(self._base_dir + '/train.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/test.txt', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/processed_h5/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_full = h5f['label_full'][:]
        image = image.swapaxes(0, 2)
        label = label.swapaxes(0, 2)
        label_full = label_full.swapaxes(0, 2)
        image = (image - np.mean(image)) / np.std(image)
        label[label > 0] = 1
        label_full[label_full > 0] = 1
        sample = {'image': image, 'label': label.astype(np.uint8), 'label_full': label_full.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample

