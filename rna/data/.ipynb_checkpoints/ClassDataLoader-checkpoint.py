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
    _models_dir = 'models'
    _data_dir = 'data'
    _samples_dir = 'samples'
    _repo_download_dir = '.'
    _instance = None
    _base_path = None
    _base_url = "https://api.github.com/repos/RNA-UNIV/datasets/contents"
    _models_path = None
    _data_path = None
    _samples_path = None

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
        cls._models_path = os.path.join(cls._base_path, cls._models_dir)
        cls._data_path = os.path.join(cls._base_path, cls._data_dir)
        cls._samples_path = os.path.join(cls._base_path, cls._samples_dir)
        os.makedirs(cls._models_path, exist_ok=True)
        os.makedirs(cls._data_path, exist_ok=True)
        os.makedirs(cls._samples_path, exist_ok=True)

    @classmethod
    def _list_files(cls, subfolder, filetype=['file', 'dir']):
        url = f"{cls._base_url}/{subfolder}"
        response = requests.get(url)
        if response.status_code == 200:
            files = response.json()
            return [file['name'] for file in files if file['type'] in filetype]
        else:
            return []

    @classmethod
    def list_datasets(cls):
        return cls._list_files(f"{cls._repo_download_dir}/{cls._data_dir}", filetype=['dir'])

    @classmethod
    def _download_file(cls, url, local_path, verbose=1):
        if verbose:
            session = requests.Session()
            response = session.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 2048
            filename = url.split('/')[-1]

            progress_bar = tqdm(total=total_size_in_bytes, unit=' iB', unit_scale=True, desc=f'Descargando {filename}')

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
    def _download_repo_directory(cls, github_path, local_path, force=False, verbose=1):
        url = f"{cls._base_url}/{github_path}"
        response = requests.get(url)
        response.raise_for_status()
        contents = response.json()

        for item in contents:
            if item['type'] == 'file':
                file_url = item['download_url']
                file_path = os.path.join(local_path, item['name'])
                if force or not os.path.exists(file_path):
                    cls._download_file(file_url, file_path, not file_path.endswith('.json'))
            elif item['type'] == 'dir':
                new_local_path = os.path.join(local_path, item['name'])
                os.makedirs(new_local_path, exist_ok=True)
                cls._download_repo_directory(item['path'], new_local_path, force, verbose)

    @classmethod
    def load_data(cls, github_path, local_subpath, force=False, verbose=1):
        local_path = os.path.join(cls._base_path, local_subpath)
        os.makedirs(local_path, exist_ok=True)
        cls._download_repo_directory(github_path, local_path, force, verbose)

    @classmethod
    def load_dataframe(cls, nombre, encoding=None, separator=None):
        (local_path, files) = cls._require_repo_directory(nombre)

        file_path = os.path.join(local_path, files[0])
        if encoding is None:
            encoding = cls._detect_encoding(file_path)
        if separator is None:
            separator = cls._detect_separator(file_path, encoding)

        df = pd.read_csv(file_path, encoding=encoding, sep=separator)
        return df

    @classmethod
    def load_array(cls, nombre, encoding=None, separator=None):
        df = cls.load_dataframe(nombre, encoding, separator)
        return (df.columns, df.to_numpy())

    @classmethod
    def _require_repo_directory(cls, nombre):
        nombre = nombre.lower()
        local_path = os.path.join(cls._data_path, nombre)
        github_path = f"{cls._repo_download_dir}/{cls._data_dir}/{nombre}"

        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
            cls._download_repo_directory(github_path, local_path)
        # agregar que si no coinciden los archivos en las carpetas, descargarlos
        files = [f for f in os.listdir(local_path) if not f.endswith('.json')]
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos de datos en la carpeta {local_path}")

        if files[0].endswith('.zip'):
            zip_path = os.path.join(local_path, files[0])
            #with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            #    zip_ref.extractall(local_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Obtener la lista de archivos dentro del ZIP
                zip_files = zip_ref.namelist()
                # Descomprimir con barra de progreso
                for file in tqdm(zip_files, desc="Descomprimiendo", unit=" archivo"):
                    zip_ref.extract(file, local_path)

        return (local_path, files)      
        
    @classmethod
    def load_images(cls, nombre, resize=None):
        (local_path, files) = cls._require_repo_directory(nombre)
        
        images = []
        labels = []
        
        # Obtener las subcarpetas que representan las clases
        subdirectories = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
        
        # Generar una lista de todas las imágenes y sus etiquetas
        all_image_files = []
        for label, subdir in enumerate(subdirectories):
            subdir_path = os.path.join(local_path, subdir)
            image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            for image_file in image_files:
                image_path = os.path.join(subdir_path, image_file)
                all_image_files.append((image_path, label))
        
        # Cargar todas las imágenes con una única barra de progreso general
        for image_path, label in tqdm(all_image_files, desc="Cargando imágenes", unit=" imagen", mininterval=5.0):
            try:
                image = Image.open(image_path)
                if resize:
                    image = image.resize(resize, Image.ANTIALIAS)
                image_np = np.array(image)
                images.append(image_np)
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
    def dataset_info(cls, nombre):
        nombre = nombre.lower()
        (local_path, files) = cls._require_repo_directory(nombre)

        info_file_path = os.path.join(local_path, 'info.json')
        if not os.path.exists(info_file_path):
            raise FileNotFoundError(f"No se encontró información sobre el dataset \"{nombre}\"")

        with open(info_file_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        return info_data

    @classmethod
    def dataset_path(cls, nombre):
        (local_path, files) = cls._require_repo_directory(nombre)

        return local_path