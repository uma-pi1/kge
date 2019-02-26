import torch


class BaseDataView(torch.utils.data.Dataset):
    """ 
    Prepares data so user-chosen loss function can be used in training 
    Data manipulations shared by all goes here, e.g. reciprocal trick
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def get_number_entities(self):
        raise NotImplementedError

    def get_number_relations(self):
        raise NotImplementedError


class NegSamplingDataView(BaseDataView):
    """  """


class OneToNDataView(BaseDataView):
    """  """


class NToNDataView(BaseDataView):
    """  """
