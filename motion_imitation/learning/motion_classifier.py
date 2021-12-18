import torch
import torch.nn as nn

class MotionClassifier(nn.Module):
    """
    The input is the info of action, and ouput is the motion id.
    This Net is used to classify current action is which motion.

    The loss function is cross entropy and regularization term(multi_motions_loss)
    """
    def __init__(self, in_dim= 12, motion_num= 2):
        super(MotionClassifier, self).__init__()

        self._in_dim = in_dim
        self._motion_num = motion_num

        self.classifier = nn.Sequential(
            nn.Linear(self._in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self._motion_num),
            nn.Softmax()
        )

    def forward(self, action):
        return self.classifier(action)

    def optimizer(self, lr =1e-3, parameters=None):
        return torch.optim.SGD(parameters, lr)

