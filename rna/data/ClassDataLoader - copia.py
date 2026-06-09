import os
import requests
import pandas as pd
import json
import zipfile
import numpy as np
from chardet.universaldetector import UniversalDetector
from tqdm import tqdm
from PIL import Image

import os
import requests
import pandas as pd
import json
import zipfile
import numpy as np
from chardet.universaldetector import UniversalDetector
from tqdm import tqdm
from PIL import Image


class DataLoader:
    _resource_dir = 'data'
    _repo_files = None
    _instance = None
    _base_path = None
    _base_url = "https://api.github.com/repos/RNA-UNIV/datasets/contents"
    _raw_base_url = "https://raw.githubusercontent.com/RNA-UNIV/datasets/main"
    _resource_path = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataLoader, cls).__new__(cls, *args, **kwargs)
            cls._base_path = os.path.join(os.getcwd(), '..', 'rna_downloads')
            cls._create_directories()
        return cls._instance

    @classmethod
    def _create_directories(cls):
        if not os.path.exists(cls._base_path):
            os.makedirs(cls._base_path)
        cls._resource_path = os.path.join(cls._base_path, cls._resource_dir)
        os.makedirs(cls._resource_path, exist_ok=True)

    @classmethod
    def _list_files(cls, subfolder='', filetype=['file', 'dir']):
        url = f"{cls._base_url}/{subfolder}"
        response = requests.get(url)
        if response.status_code == 200:
            files = response.json()
            return [file['name'] for file in files if file['type'] in filetype]
        else:
            return []

    @classmethod
    def list_datasets(cls):
        if cls._repo_files is None:
            cls._repo_files = cls._list_files(filetype=['dir'])
        return cls._repo_files

    @classmethod
    def _download_file(cls, url, local_path, verbose=True):
        if verbose:
            session = requests.Session()
            response = session.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 2048
            filename = url.split('/')[-1]
            progress_bar = tqdm(total=total_size_in_bytes, unit=' iB', unit_scale=True,
                                desc=f'Descargando {filename}')
            with open(local_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes > 0 and progress_bar.n < total_size_in_bytes:
                print("Sucedió un error al descargar")
        else:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as file:
                file.write(response.content)

    @classmethod
    def _extract_zip(cls, zip_path, dest_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_files = zip_ref.namelist()
            for file in tqdm(zip_files, desc="Descomprimiendo", unit=" archivo"):
                zip_ref.extract(file, dest_path)
        os.remove(zip_path)

    @classmethod
    def _require_repo_directory(cls, name, force=False):
        name = name.lower()
        local_path = os.path.join(cls._resource_path, name)
        os.makedirs(local_path, exist_ok=True)

        sentinel = os.path.join(local_path, '.complete')
        if not force and os.path.exists(sentinel):
            return local_path

        expected_files = cls._list_files(subfolder=name, filetype=['file'])
        if not expected_files:
            raise FileNotFoundError(f"No se encontró el dataset \"{name}\" en el repositorio")

        missing_files = [f for f in expected_files
                         if force or not os.path.exists(os.path.join(local_path, f))]

        for i, filename in enumerate(missing_files, start=1):
            local_file = os.path.join(local_path, filename)
            url = f"{cls._raw_base_url}/{name}/{filename}"
            print(f"[{i}/{len(missing_files)}] {filename}")
            cls._download_file(url, local_file,
                               verbose=not filename.endswith(('.json', '.md')))

        zip_files = [f for f in os.listdir(local_path) if f.endswith('.zip')]
        for zip_file in zip_files:
            zip_path = os.path.join(local_path, zip_file)
            data_path = os.path.join(local_path, 'data')
            os.makedirs(data_path, exist_ok=True)
            cls._extract_zip(zip_path, data_path)

        open(sentinel, 'w').close()
        return local_path

    @classmethod
    def dataset_path(cls, name):
        return cls._require_repo_directory(name)

    @classmethod
    def load_dataframe(cls, name, encoding=None, separator=None):
        local_path = cls._require_repo_directory(name)

        csv_files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No se encontró archivo CSV en {local_path}")

        file_path = os.path.join(local_path, csv_files[0])
        if encoding is None:
            encoding = cls._detect_encoding(file_path)
        if separator is None:
            separator = cls._detect_separator(file_path, encoding)

        return pd.read_csv(file_path, encoding=encoding, sep=separator)

    @classmethod
    def load_array(cls, name, encoding=None, separator=None):
        df = cls.load_dataframe(name, encoding, separator)
        return (df.columns, df.to_numpy())

    @classmethod
    def load_images(cls, name, resize=None):
        local_path = cls._require_repo_directory(name)
        data_path = os.path.join(local_path, 'data')
        search_path = data_path if os.path.exists(data_path) else local_path

        images, labels = [], []
        subdirectories = [d for d in os.listdir(search_path)
                          if os.path.isdir(os.path.join(search_path, d))]

        all_image_files = []
        for label, subdir in enumerate(subdirectories):
            subdir_path = os.path.join(search_path, subdir)
            for f in os.listdir(subdir_path):
                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    all_image_files.append((os.path.join(subdir_path, f), label))

        for image_path, label in tqdm(all_image_files, desc="Cargando imágenes",
                                      unit=" imagen", mininterval=5.0):
            try:
                image = Image.open(image_path)
                if resize:
                    image = image.resize(resize, Image.ANTIALIAS)
                images.append(np.array(image))
                labels.append(label)
            except Exception as e:
                print(f"Error al cargar la imagen: {image_path}, {e}")

        return np.array(images), np.array(labels)

    @classmethod
    def _detect_encoding(cls, file_path):
        detector = UniversalDetector()
        with open(file_path, 'rb') as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
        detector.close()
        return detector.result['encoding']

    @classmethod
    def _detect_separator(cls, file_path, encoding):
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
            separators = [',', ';', '\t']
            return max(separators, key=lambda sep: first_line.count(sep))

    @classmethod
    def dataset_info(cls, name, force=False):
        name = name.lower()
        local_path = os.path.join(cls._resource_path, name)
        os.makedirs(local_path, exist_ok=True)

        info_file = os.path.join(local_path, 'info.json')
        if force or not os.path.exists(info_file):
            url = f"{cls._raw_base_url}/{name}/info.json"
            cls._download_file(url, info_file, verbose=False)

        if not os.path.exists(info_file):
            raise FileNotFoundError(f"No se encontró información sobre el dataset \"{name}\"")

        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def dataset_info_display(cls, name, force=False):
        info_data = cls.dataset_info(name, force)

        try:
            from IPython import get_ipython
            from IPython.display import display, JSON
            if get_ipython() is not None:
                display(JSON(info_data, root=name))
            else:
                raise RuntimeError
        except (ImportError, RuntimeError):
            print(json.dumps(info_data, indent=2, ensure_ascii=False))


