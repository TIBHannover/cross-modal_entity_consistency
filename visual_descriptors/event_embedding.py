import imageio
import logging
import numpy as np
import os
import torch
import torchvision


class EventClassificator(torch.nn.Module):
    def __init__(self, model_path, use_cpu=False):

        if not use_cpu and torch.cuda.is_available():
            self._device = "cuda:0"
        else:
            self._device = "cpu"

        logging.info(f"Using {self._device}")

        super(EventClassificator, self).__init__()

        resnet_model = torchvision.models.resnet.resnet50(pretrained=False)

        self._features = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        self._fc = torch.nn.Linear(2048, 409)  # model trained for an ontology with 409 nodes or 148 classes

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.eval()
        self.to(self._device)
        self._load(os.path.join(model_path, 'models', 'VisE_CO_cos.pt'), device=self._device)

    def get_img_embedding(self, img_path):
        try:
            img = imageio.imread(img_path)

            if len(img.shape) == 2:
                img = np.stack([img] * 3)

            if len(img.shape) == 3:
                if img.shape[-1] > 3:
                    img = img[..., 0:3]
                if img.shape[-1] < 3:
                    img = np.stack([img[..., 0]] * 3)

            if self._transform:
                img = self._transform(img)

            # add batch dimension
            img = np.expand_dims(img, axis=0)

            img = torch.tensor(img).to(self._device)
            x = self._features(img)
            x = torch.flatten(x, 1)

            return [x.detach().cpu().numpy().squeeze()]  # return as list for compatability to face verification

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f'Cannot create embedding for {img_path} for reason {e}')
            return []

    def _load(self, checkpoint_path, device="cpu"):

        state_dict = torch.load(checkpoint_path, map_location=device)

        try:
            self.load_state_dict(state_dict["model"])
        except RuntimeError:
            logging.warn("Trainer: Save DataParallel model without using module")
            map_dict = {}
            for key, value in state_dict["model"].items():
                if key.split(".")[0] == "module":
                    map_dict[".".join(key.split(".")[1:])] = value
                else:
                    map_dict["module." + key] = value
            self.load_state_dict(map_dict)