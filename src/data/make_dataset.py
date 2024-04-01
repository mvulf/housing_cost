import os
from pathlib import Path

# import dload
# from googledriver import download
# URL = 'https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view?usp=share_link'
# URL_bypass = 'https://drive.google.com/file/d/12NmSGbXWPQbuKNwEtFo0hx4J7NOLNCG2/view?usp=sharing'

raw_dir = '../../data/raw/'
raw_file = 'data.csv'
raw_path = Path(raw_dir, raw_file)

if __name__ == '__main__':
    is_file = os.path.isfile(raw_path)
    if is_file:
        print(f'Файл {raw_path} присутствует')
    else:
        print(f'Файл {raw_path} отсутствует. Убедитесь в корректности пути и что он загружен средствами DVC.')
    # try:
    #     download(URL, save_path, 'data')
    #     # dload.save_unzip("https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view?usp=share_link", "../../data/raw")
    # except:
    #     download(URL_bypass, save_path, 'data')
    #     # dload.save_unzip("https://drive.google.com/file/d/12NmSGbXWPQbuKNwEtFo0hx4J7NOLNCG2/view?usp=sharing", "../../data/raw")
    
    