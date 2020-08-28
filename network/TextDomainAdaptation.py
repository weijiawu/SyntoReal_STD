import torch
import torch.nn.functional as F
from torch import nn
from network.gradient_scalar_layer import GradientScalarLayer


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        return self.conv2_da(t)


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(2028, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self):
        super(DomainAdaptationModule, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid_gap = nn.Sigmoid()

        self.grl_img = GradientScalarLayer(-1.0 * 0.1)
        self.grl_ins = GradientScalarLayer(-1.0 * 0.1)
        self.grl_img_consist = GradientScalarLayer(1.0 * 0.1)
        self.grl_ins_consist = GradientScalarLayer(1.0 * 0.1)

        self.imghead = DAImgHead()
        self.inshead = DAInsHead()


    def forward(self, img_features):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        img_grl_fea = self.grl_img(self.gap(img_features))  # 经过grl的image-level feature

        da_img_features = self.sigmoid_gap(self.imghead(img_grl_fea))  # 对image-level feature的每个层都压成1*1

        return da_img_features