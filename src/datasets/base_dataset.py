from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, args=None, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.samples = []
        self.imgnames = []
        # Add any other common initializations here

    def __len__(self):
        if hasattr(self, 'subsampled_keys'):
            return len(self.subsampled_keys)
        return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, 'samples') and self.samples:
            seqName, index = self.samples[idx]
            return self.getitem(seqName, index)
        raise NotImplementedError("BaseDataset requires 'samples' to be set.")

    def getitem(self, seqName, index):
        """
        Generic getitem method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("getitem must be implemented in child dataset class.")

    def get_img_data(self, imgname, load_rgb=True):
        """
        Generic get_img_data method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("get_img_data must be implemented in child dataset class.")

    def get_future_data(self, imgname, indices):
        """
        Generic get_future_data method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("get_future_data must be implemented in child dataset class.")

    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        """
        Generic get_fixed_length_sequence method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("get_fixed_length_sequence must be implemented in child dataset class.")

    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        """
        Generic get_variable_length_sequence method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("get_variable_length_sequence must be implemented in child dataset class.")

    def get_imgname_from_index(self, seqname, index):
        """
        Generic get_imgname_from_index method. Should be overridden by child classes for specific logic.
        """
        raise NotImplementedError("get_imgname_from_index must be implemented in child dataset class.")
