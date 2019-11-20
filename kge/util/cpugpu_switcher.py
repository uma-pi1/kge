import torch
import numpy as np

"""
This cpu_gpu switcher very much inspired by SpeedTorch (MIT licensed):
https://github.com/Santosh-Gupta/SpeedTorch
"""


class SwitcherBase:
    """
    This class handles the mapping of indexes between two tensors.
    """
    def __init__(self, num_embeddings, dimension, device="cuda"):
        self.num_embeddings = num_embeddings
        self.dimension = dimension
        self.indexes = None
        self.mapped_indexes = None
        self.mapper = None
        self.device = device

    def set_indexes(self, index):
        """
        set the indexes for the tensor on CPU
        :param index: one dimensional tensor containing indexes for CPU tensor
        :return: None
        """
        self.indexes = index.to("cpu").long()

    def set_mapped_index(self, mapped_index):
        """
        set the mapped indexes for tensor on GPU
        :param mapped_index: one dimensional tensor containing mapped indexes for GPU
        :return: None
        """
        self.mapped_indexes = mapped_index.to(self.device).long()

    def create_indexes(self, indexes, device="cuda"):
        """
        create indexes to map the ids of the entities to the ids corresponding to the smaller tensors on GPU
        :param indexes: one dimensional tensor with the indexes to map.
        :param device: cuda device
        :return: None
        """
        self.indexes = indexes.to("cpu").long()
        self.mapped_indexes = torch.arange(len(indexes))
        self.mapper = dict(zip(self.indexes.numpy(), self.mapped_indexes.cpu().numpy()))
        self.mapped_indexes = self.mapped_indexes.to(device).long()

    def map_indexes(self, values, device="cuda"):
        """
        map ids of the entities to the ids corresponding to the smaller tensors on GPU
        :param values: one dimensional tensor containing ids to map
        :param device: cuda device
        :return: one dimensional tensor containing the mapped indexes
        """
        mapped = np.vectorize(self.mapper.get)(values.cpu().numpy())
        #mapped = values.cpu().apply_(lambda x: self.mapper[x])
        return torch.LongTensor(mapped).to(device)


class CPUEmbeddingSwitcher(SwitcherBase):
    """
    This class handles the switching between the currently on GPU needed embeddings and the embedding tensor on CPU.
    """
    def __init__(self, gpu_embedding_layer, num_embeddings, dimension, device="cuda"):
        super().__init__(num_embeddings, dimension, device)
        self.gpu_embedding_layer = gpu_embedding_layer
        self.cpu_tensor = torch.FloatTensor(self.num_embeddings, self.dimension)
        if self.device.startswith("cuda"):
            self.cpu_tensor.pin_memory()

    def to_gpu(self):
        self.gpu_embedding_layer.weight.data[self.mapped_indexes, :] = self.cpu_tensor[self.indexes, :].to(self.device)

    def to_cpu(self):
        self.cpu_tensor[self.indexes, :] = self.gpu_embedding_layer.weight.data[self.mapped_indexes, :]\
            .detach().to("cpu").pin_memory()

    def load(self, savepoint):
        self.cpu_tensor = savepoint
        if self.device.startswith("cuda"):
            self.cpu_tensor.pin_memory()


class CPUOptimizerSwitcher(SwitcherBase):
    """
    This class handles the switching of the parts of the optimizer corresponding to the currently on GPU needed
    embeddings and the complete optimizer tensors on CPU
    """
    def __init__(self, optimizer, num_embeddings, dimension, model, device="cuda"):
        super().__init__(num_embeddings, dimension, device)
        self.optimizer = optimizer
        self.model = model
        self.variable_name = "_entity_embedder.embeddings"
        self.optimizer_variable_list = self._initialize_optimizer_state()
        self.optimizer_key = self._get_optimizer_key()
        self.cpu_var = []
        self._initialize_tensors()
        self._pin_tensors()

    def _get_optimizer_key(self):
        """
        get the optimizer key for the layer/tensor to optimize
        :return: optimizer key
        """
        optimizer_index = None
        for i, item in enumerate(self.model.named_parameters()):
            if item[0] == f"{self.variable_name}.weight":
                optimizer_index = i
        if optimizer_index is None:
            print("Error: No variable with that name is in Model. Please initialize again with correct name")
            exit(1)

        optimizer_key_list = list(self.optimizer.state_dict()["state"].keys())
        optimizer_key = optimizer_key_list[optimizer_index]
        return optimizer_key

    def _initialize_optimizer_state(self):
        """
        Some optimizers do not initialize its state until after the first step.
        Therefore initialize them here
        :return: list of optimizer variables
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                # State initialization

                if self.optimizer.__str__().split(" ", 1)[0] == "Adagrad":
                    optimizer_variable_list = ["sum"]
                    return optimizer_variable_list
                else:
                    print("this optimizer is currently not supported")
                    exit(1)

    def _initialize_tensors(self):
        """
        Initialize the tensors for the optimizer variables on cpu
        :return: None
        """
        for i in range(len(self.optimizer_variable_list)):
            self.cpu_var.append(torch.zeros(size=(self.num_embeddings, self.dimension),
                                            dtype=torch.float, device="cpu"))

    def _pin_tensors(self):
        if self.device.startswith("cuda"):
            for tensor in self.cpu_var:
                tensor.pin_memory()

    def to_gpu(self):
        for idx, optimizer_var in enumerate(self.optimizer_variable_list):
            self.optimizer.state_dict()["state"][self.optimizer_key][optimizer_var][self.mapped_indexes, :] = \
                self.cpu_var[idx][self.indexes, :].to(self.device)

    def to_cpu(self):
        for idx, optimizer_var in enumerate(self.optimizer_variable_list):
            self.cpu_var[idx][self.indexes, :] = self.optimizer.state_dict()["state"][self.optimizer_key][
                                                     optimizer_var][self.mapped_indexes, :].detach().cpu().pin_memory()

    def load(self, savepoint):
        self.cpu_var = savepoint
        self._pin_tensors()
