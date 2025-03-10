import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import sys
sys.path.insert(0, "/home/jupyter/LLaVA")
from ecg_coca.training import get_ecg_encoder

class CLIPECGTower(nn.Module):
    def __init__(self, ecg_tower, args, delay_load=False):
        super().__init__()

        self.model_config = None
        self.ecg_processor = None
        self.ecg_tower = None
        self.ecg_tower_is_loaded = False

        self.ecg_tower_name = ecg_tower
        self.model_name = getattr(args, 'open_clip_config', None)
        if self.model_name is None:
            raise ValueError('No open_clip config for building ECG encoder!')
        self.load_model(self.model_name)
        # if not delay_load:
        #     self.load_model(self.model_name)
        # elif getattr(args, 'unfreeze_mm_vision_tower', False):
        #     self.load_model(self.model_name)

        ecg_config = self.model_config.get('ecg_cfg', {})

        self.hidden_size = ecg_config.get('width', 768)
        self.seq_length = ecg_config.get('seq_length', 5000)
        self.patch_size = ecg_config.get('patch_size', 50)
        self.device = self.ecg_tower.state_dict()['class_embedding'].device
        self.dtype = self.ecg_tower.state_dict()['class_embedding'].dtype

        self.num_patches_per_side = self.seq_length // self.patch_size
        self.num_patches = self.seq_length // self.patch_size

    def is_loaded(self):
        return self.ecg_tower_is_loaded

    def load_model(self, model_name, device_map=None):
        if self.ecg_tower_is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.ecg_tower, self.ecg_processor, self.model_config = get_ecg_encoder(model_name, checkpoint_path=self.ecg_tower_name, device='cpu')
        self.ecg_tower.requires_grad_(False)

        self.ecg_tower_is_loaded = True
        print("Loaded {} model".format(self.ecg_tower_name))

    @torch.no_grad()
    def forward(self, ecgs):
        self.device = self.ecg_tower.state_dict()['class_embedding'].device
        self.dtype = self.ecg_tower.state_dict()['class_embedding'].dtype
        if type(ecgs) is list:
            ecg_features = []
            for ecg in ecgs:
                ecg_feature = self.ecg_tower(ecg.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_last_transformer_layer=True)
                ecg_feature = ecg_feature.to(ecg.dtype)
                ecg_features.append(ecg_feature)
        else:
            ecg_features = self.ecg_tower(ecgs.to(device=self.device, dtype=self.dtype), output_last_transformer_layer=True)
            ecg_features = ecg_features.to(ecgs.dtype)

        return ecg_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
        
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        
    def is_loaded(self):
        return self.is_loaded
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
