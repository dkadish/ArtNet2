import torch
import torchvision
from torch.utils import model_zoo
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def fasterrcnn_shape_resnet50(device, train_dataset_size, batch_size_train, num_epochs, trainable_layers,
                              box_nms_thresh, num_classes):

    #TODO Testing a new method for this in ArtNet2_peopleart_and_voc_training_with_SIN_backbone.ipynb. Wait for results.

    # url_resnet50_trained_on_SIN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'
    url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'

    backbone_model = torchvision.models.resnet50(pretrained=False).cuda()
    checkpoint = model_zoo.load_url(url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN)

    # Some magic to rename the keys so that it doens't have to be parallelized
    state_dict = dict([('.'.join(k.split('.')[1:]), v) for k, v in checkpoint["state_dict"].items()])

    backbone_model.load_state_dict(state_dict)

    # load a pre-trained model for classification and return
    # only the features
    # features = list(torchvision.models.resnet101(pretrained=True).children())[:-3]
    features = list(backbone_model.children())[:-3]
    backbone = torch.nn.Sequential(*features)

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here.
    # For resnet101, I THINK it is 1000
    backbone.out_channels = 1024

    for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        for parameter in parameters:
            parameter.requires_grad = False

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # # move model to the right device
    model.to(device)

    # # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # # and a learning rate scheduler which decreases the learning rate by
    # # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)

    return model, optimizer


def fasterrcnn_resnet101(device, trainable_layers, box_nms_thresh, num_classes):
    return fasterrcnn_resnetx('resnet101', device, trainable_layers, box_nms_thresh, num_classes)


def fasterrcnn_resnet50(device, trainable_layers, box_nms_thresh, num_classes):
    return fasterrcnn_resnetx('resnet50', device, trainable_layers, box_nms_thresh, num_classes)


def fasterrcnn_resnetx(backbone_name, device, trainable_layers,
                       box_nms_thresh,
                       num_classes):

    backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=trainable_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, box_nms_thresh=box_nms_thresh)
    model.to(device)

    params_to_update = model.parameters()
    optimizer = torch.optim.SGD(params_to_update, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)

    return model, optimizer
