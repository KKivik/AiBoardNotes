from datasets import load_dataset

# Указываем путь, куда именно скачать файлы
dataset = load_dataset("deepcopy/MathWriting-human", cache_dir="./my_dataset_folder")