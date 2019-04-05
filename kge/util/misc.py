import subprocess
import torch


class KgeLoss:
    """ Wraps torch loss functions """

    def create(config):
        """ Factory method for loss creation """
        if config.get('train.loss') == 'ce':
            return torch.nn.CrossEntropyLoss(reduction='mean')
        elif config.get('train.loss') == 'bce':
            return torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif config.get('train.loss') == 'kl':
            return torch.nn.KLDivLoss(reduction='mean')
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError('train.loss')

def is_number(s, number_type):
    """ Returns True is string is a number. """
    try:
        number_type(s)
        return True
    except ValueError:
        return False


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()