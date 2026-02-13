from datasets import load_dataset
from torchvision import transforms
from dotenv import load_dotenv
import os

load_dotenv()

# путь при скачивании
cache_dir = "./my_dataset_folder"
print("загрузка...")
# библиотека проверит папку и мгновенно загрузит данные из кэша
dataset = load_dataset("deepcopy/MathWriting-human", cache_dir=cache_dir)
print("загружено успешно")

transform = transforms.Compose([
    transforms.ColorJitter(contrast=(5, 5)),  # >0 повышает яркость
    transforms.Grayscale(num_output_channels=1),  # [H, W] → 1 канал
    transforms.ToTensor(),                        # [1, H, W], float32 [0,1]
])

tr = dataset["train"]

images = []
latex = []

for i in range(len(tr)):
    cur = tr[i]
    im = transform(cur["image"])
    txt = cur["latex"]
    images.append(im)
    latex.append()


