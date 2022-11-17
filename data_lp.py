import utils
import random
import torch as th
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


preprocess = _transform(224)


class LinearProbeDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a tensor representing image after transformation
    """

    def __init__(self, cls2img, processed_path, n_shots, cls_names, img_ext='jpg'):
        self.data_path = processed_path
        self.images = []
        self.labels = []
        self.img_ext = img_ext
        for cls_name, imgs in cls2img.items():
            if cls_name not in cls_names:
                continue
            label = cls_names.index(cls_name)
            if n_shots != 'all':
                imgs = random.sample(imgs, n_shots)
            self.images.extend(imgs)
            self.labels += [label] * len(imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = preprocess(
            Image.open(
                self.data_path.joinpath('{}{}'.format(self.images[idx],
                                                       self.img_ext))))
        return image, self.labels[idx]


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root,
                 batch_size,
                 data_split_path,
                 img_path,
                 n_shots,
                 cls_names,
                 num_workers=0,
                 img_ext='.jpg'):
        super().__init__()
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.bs = batch_size
        self.num_workers = num_workers
        self.cls2train = utils.pickle_load(
            Path(data_split_path).joinpath('class2images_train.p'))
        self.cls2val = utils.pickle_load(
            Path(data_split_path).joinpath('class2images_val.p'))
        self.cls2test = utils.pickle_load(
            Path(data_split_path).joinpath('class2images_test.p'))
        self.data_split_path = data_split_path
        self.img_path = Path(img_path)
        self.cls_names = cls_names
        self.img_ext = img_ext
        self.n_shots = n_shots

    def compute_img_feat(self, cls2img, n_shots):
        labels = []
        all_img_paths = []
        for i, (cls_name, img_names) in enumerate(cls2img.items()):
            if n_shots !=  'all':
                img_names = img_names[:n_shots]
            labels.extend([self.cls_names.index(cls_name)] * len(img_names))
            all_img_paths.extend(
                [self.img_path.joinpath('{}{}'.format(img_name, self.img_ext))\
                     for img_name in img_names])

        img_feat = utils.prepare_img_feat(all_img_paths)
        return img_feat, th.tensor(labels)

    def prepare_img_feat_for_splits(self):
        train_img_data_path = self.data_root.joinpath('train_img_data.pth')
        val_img_data_path = self.data_root.joinpath('val_img_data.pth')
        test_img_data_path = self.data_root.joinpath('test_img_data.pth')
        if not train_img_data_path.exists():
            img_feat_train, label_train = self.compute_img_feat(self.cls2train, self.n_shots)

            th.save({
                "img_feat": img_feat_train,
                "label": label_train
            }, train_img_data_path)
        else:
            img_data = th.load(train_img_data_path)
            img_feat_train, label_train = img_data['img_feat'], img_data[
                'label']
        if not val_img_data_path.exists():
            img_feat_val, label_val = self.compute_img_feat(self.cls2val, self.n_shots)
            th.save({
                "img_feat": img_feat_val,
                "label": label_val
            }, val_img_data_path)
        else:
            img_data = th.load(val_img_data_path)
            img_feat_val, label_val = img_data['img_feat'], img_data['label']
        if not test_img_data_path.exists():
            img_feat_test, label_test = self.compute_img_feat(self.cls2test, self.n_shots)
            th.save({
                "img_feat": img_feat_test,
                "label": label_test
            }, test_img_data_path)
        else:
            img_data = th.load(test_img_data_path)
            img_feat_test, label_test = img_data['img_feat'], img_data['label']
        self.img_feat_train = img_feat_train
        self.label_train = label_train
        self.img_feat_val = img_feat_val
        self.label_val = label_val
        self.img_feat_test = img_feat_test
        self.label_test = label_test

    def setup(self, stage=None):
        self.train_dataset = self.val_dataset = self.test_dataset = None
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.bs,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.bs,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()


class LinearProbeDataModule(DataModule):

    def __init__(self,
                 data_root,
                 batch_size,
                 data_split_path,
                 img_path,
                 n_shots,
                 cls_names, 
                 num_workers=0,
                 img_ext='jpg'):
        super().__init__(data_root, batch_size, data_split_path, img_path, n_shots, cls_names,
                         num_workers)
        self.n_shots = n_shots
        self.img_ext = img_ext


    def setup(self, stage=None):
        self.train_dataset = LinearProbeDataset(self.cls2train, self.img_path,
                                                self.n_shots, self.cls_names, img_ext=self.img_ext)
        self.val_dataset = LinearProbeDataset(self.cls2val, self.img_path,
                                              'all', self.cls_names, img_ext=self.img_ext)
        self.test_dataset = LinearProbeDataset(self.cls2test, self.img_path,
                                               'all', self.cls_names, img_ext=self.img_ext)