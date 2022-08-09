import torch
from .base_model import BaseModel
from . import networks


class GALNetModel(BaseModel):
    def name(self):
        return 'GALNet'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt, dataset):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['segmentation']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['rgb_image', 'tdisp_image', 'label', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['GALNet']

        # load/define networks
        if opt.input == "rgb":
            print("Using RGB images as input")
            self.input_channels = 3
        elif opt.input == "tdisp":
            print("Using transformed disparity images as input")
            self.input_channels = 1
        else:
            raise NotImplementedError

        self.netGALNet = networks.define_GALNet(dataset.num_labels, gpu_ids= self.gpu_ids, input_channels= self.input_channels, use_gal=opt.gal)
        # define loss functions
        self.criterionSegmentation = networks.SegmantationLoss(class_weights=None).to(self.device)

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []
            self.optimizer = torch.optim.SGD(self.netGALNet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.set_requires_grad(self.netGALNet, True)

    def set_input(self, input):
        self.rgb_image = input['rgb_image'].to(self.device)
        self.tdisp_image = input['tdisp_image'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_names = input['path']

    def forward(self):
        if self.opt.input == "rgb":
            self.output = self.netGALNet(self.rgb_image)
        elif self.opt.input == "tdisp":
            self.output = self.netGALNet(self.tdisp_image)
        else:
            raise NotImplementedError

    def get_loss(self):
        self.loss_segmentation = self.criterionSegmentation(self.output, self.label)

    def backward(self):
        self.loss_segmentation.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer.step()
