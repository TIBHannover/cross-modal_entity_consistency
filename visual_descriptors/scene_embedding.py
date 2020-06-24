import logging
import os
from PIL.Image import open as open_image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F


class SceneClassificator:
    def __init__(self, model_path=None):

        if model_path is not None:
            model = models.__dict__['resnet50'](num_classes=365)
            checkpoint = torch.load(os.path.join(model_path, 'resnet50_places365.pth.tar'),
                                    map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval().to(self._device)
            self.model = model

            # method for centre crop
            self._centre_crop = trn.Compose([
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            logging.warning('No model built.')

    def get_img_embedding(self, img_path):
        try:
            img = open_image(img_path).convert('RGB')
            input_img = V(self._centre_crop(img).unsqueeze(0)).to(self._device)

            # forward pass for feature extraction
            x = input_img
            i = 0
            for module in self.model._modules.values():
                if i == 9:
                    break
                x = module(x)
                i += 1

            return [x.detach().cpu().numpy().squeeze()]  # return as list for compatability to face verification
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            logging.error(f'Cannot create embedding for {img_path}')
            return []

    def get_logits(self, img_path):
        try:
            img = open_image(img_path).convert('RGB')
            input_img = V(self._centre_crop(img).unsqueeze(0)).to(self._device)

            logit = self.model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            return h_x.detach().cpu().numpy().squeeze()
        except KeyboardInterrupt:
            raise
        except:
            logging.error(f'Cannot create logits for {img_path}')
            return []
