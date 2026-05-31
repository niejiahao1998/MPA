r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.deepglobe_MPA import DatasetDeepglobeMPA
from data.deepglobe import DatasetDeepglobe
from data.isic_MPA import DatasetISICMPA
from data.isic import DatasetISIC
from data.lung_MPA import DatasetLungMPA
from data.lung import DatasetLung
from data.fss_MPA import DatasetFSSMPA
from data.fss import DatasetFSS
from data.suim_MPA import DatasetSUIMMPA
from data.suim import DatasetSUIM



class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'deepglobempa': DatasetDeepglobeMPA,
            'deepglobe': DatasetDeepglobe,
            'isicmpa': DatasetISICMPA,
            'isic': DatasetISIC,
            'lungmpa': DatasetLungMPA,
            'lung': DatasetLung,
            'fssmpa': DatasetFSSMPA,
            'fss': DatasetFSS,
            'suim': DatasetSUIM,
            'suimmpa': DatasetSUIMMPA
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=0)

        return dataloader
