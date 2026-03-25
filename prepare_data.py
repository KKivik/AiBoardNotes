from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import trange
from dotenv import load_dotenv
import os
import torch
import numpy as np
import random
from PIL import Image, ImageOps
from tokenyzer import Tokenyzer

load_dotenv()

W = int(os.getenv("W"))
H = int(os.getenv("H"))
formula_start = int(os.getenv("formula_start"))
formula_end = int(os.getenv("formula_end"))
padding = int(os.getenv("padding"))


Tkn = Tokenyzer()

class SmartResizer(torch.nn.Module):
    def forward(self, img):
        img_h, img_w = img.shape[-2:]
        if img_w > W or img_h > H:
            k = min(W / img_w, H / img_h)
            # print(k)
            resizer = transforms.Compose([
                transforms.Resize((int(k * img_h), int(k * img_w)))
            ])
            img = resizer(img)
        img_h, img_w = img.shape[-2:]
        dx = W - img_w
        dy = H - img_h
        pl = 0  # left
        pu = 0  # up
        pr = 0  # right
        pd = 0  # down
        if dx > 0:
            # Add in dim=-1
            border = random.randint(1, dx + 1)
            pl = border
            pr = dx - border
        if dy > 0:
            # Add in dim=-1
            border = random.randint(1, dy + 1)
            pd = border
            pu = dy - border
        #padder = transforms.Pad((pl, pu, pr, pd))  # left, top, right and bottom
        padder = transforms.Pad((pl, pu, pr, pd), fill=1)  # left, top, right and bottom
        img = padder(img)
        return img


class PreparedDataset(Dataset):
    def __init__(self, hf_dataset, mode="train"):
        self.dataset = hf_dataset[mode]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # [H, W] → 1 канал
            transforms.Lambda(lambda x: ImageOps.invert(x)),
            transforms.ColorJitter(contrast=(5, 5)),  # >0 повышает яркость
            transforms.ToTensor(),  # [1, H, W], float32 [0,1]
            SmartResizer(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        # [301] - formula_start; [302] formula_end; [303] - padding
        self.formula_start = 301
        self.formula_end = 302
        self.padding = 303
        self.service_tokens = set([self.formula_start, self.formula_end, self.padding])
        self.max_len = 190  # + под BOS и EOS

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample["image"])

        unfold = torch.nn.Unfold(kernel_size=20, stride=20, padding=0)
        image = image.unsqueeze(1) # for torch.nn.Unfold
        image = unfold(image) # cut on 20x20 patches each sample
        image = image.permute(0, 2, 1)

        latex = Tkn.encode(sample["latex"])
        cut_latex = latex[:self.max_len - 2]
        cut_latex = [self.formula_start] + cut_latex + [self.formula_end]

        latex_x = cut_latex[:-1]
        latex_y = cut_latex[1:]

        # pad_len = self.max_len - len(latex_x) - 1
        # latex_x = latex_x + [self.padding] * pad_len
        # latex_y = latex_y + [self.padding] * pad_len

        return [image.squeeze(0), latex_x, latex_y] # (X:image, latex_x:latex_in, latex_y: target)

def padding_in_batch(batch):
    image_batch, latex_x_bat, latex_y_batch = zip(*batch) # (im1, im2, ..), (latex_x_1, latex_x_..), (latex_y_1, latex_y_2,...)
    max_len = -1
    for i in range(len(batch)):
        latex_x_temp = batch[i][1]
        max_len = max(max_len, len(latex_x_temp))

    image_out = torch.stack(list(image_batch))
    latex_x_out = torch.zeros((len(batch), max_len), dtype=torch.long)
    latex_y_out = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i in range(len(batch)):
        latex_x = batch[i][1]
        latex_y = batch[i][2]
        pad_len = max_len - len(latex_x)
        cur_lat_x = torch.tensor(latex_x + [padding] * pad_len)
        cur_lat_y = torch.tensor(latex_y + [padding] * pad_len)
        # torch.stack((latex_x_out, cur_lat_x), 0)
        # torch.stack((latex_y_out, cur_lat_y), 0)
        latex_x_out[i] = cur_lat_x
        latex_y_out[i] = cur_lat_y

    #
    # for i in range(len(batch)):
    #     latex_x = batch[i][1]
    #     latex_y = batch[i][2]
    #     pad_len = max_len - len(latex_x) - 1
    #     batch[i][1] = torch.tensor(latex_x + [padding] * pad_len)
    #     batch[i][2] = torch.tensor(latex_y + [padding] * pad_len)
    return image_out, latex_x_out, latex_y_out




# путь при скачивании
cache_dir = "./my_dataset_folder"
print("загрузка датасета MathWriting-human...")
# библиотека проверит папку и мгновенно загрузит данные из кэша
dataset = load_dataset("deepcopy/MathWriting-human", cache_dir=cache_dir)
print("датасет MathWriting-human загружен успешно")

ds_train = PreparedDataset(dataset, mode="train")
ds_test = PreparedDataset(dataset, mode="test")

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, collate_fn=padding_in_batch)
# dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=64, shuffle=True)