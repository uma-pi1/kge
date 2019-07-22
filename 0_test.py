# import torch
# torch.cuda.is_available()
# torch.cuda.current_device()  # fails here

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())