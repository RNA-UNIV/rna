import io
import os
import json
import shutil
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from chardet.universaldetector import UniversalDetector
from PIL import Image
from tqdm import tqdm  # usado en load_files


class DataLoader:
    _resource_dir = 'data'
    _repo_files = None
    _instance = None
    _base_path = None
    _base_url = "https://api.github.com/repos/RNA-UNIV/datasets/contents"
    _raw_base_url = "https://raw.githubusercontent.com/RNA-UNIV/datasets/main"
    _resource_path = None

    # Extensiones soportadas por tipo
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataLoader, cls).__new__(cls, *args, **kwargs)
            cls._base_path = os.path.join(os.getcwd(), '..', 'rna_downloads')
            cls._create_directories()
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Infraestructura interna                                             #
    # ------------------------------------------------------------------ #

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
        return []

    @classmethod
    def _print_progress(cls, prefix, msg):
        """Imprime en la misma línea sobreescribiendo la anterior."""
        line = f'{prefix}  {msg}'
        print(f'{line:<80}', end='', flush=True)

    @classmethod
    def _download_file(cls, url, local_path, verbose=True, prefix=''):
        if verbose:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=65536):
                    file.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded / total * 100) if total else 0
                    cls._print_progress(prefix, f'Descargando {pct}%')
        else:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as file:
                file.write(response.content)

    @classmethod
    def _find_tool(cls, name):
        """Busca un ejecutable via subprocess para evitar problemas de PATH en algunos entornos."""
        # shutil.which puede fallar en Colab si el PATH del proceso Python es reducido
        try:
            result = subprocess.run(
                ['which', name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            # 'which' no existe (Windows) — intentar ejecutar el comando directamente
            try:
                subprocess.run([name, '--help'], capture_output=True)
                return True
            except FileNotFoundError:
                return False

    @classmethod
    def _extract_zip(cls, zip_path, dest_path, prefix=''):
        """
        Extrae zip_path en dest_path usando el descompresor más rápido disponible.
        Orden de preferencia: unzip (Linux/Mac) → zipfile Python (fallback universal).
        tar se excluye porque no soporta formato ZIP.
        """
        import threading

        os.makedirs(dest_path, exist_ok=True)
        t0 = time.time()

        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())

        stop_event = threading.Event()

        def _progress():
            while not stop_event.is_set():
                current = sum(len(files) for _, _, files in os.walk(dest_path))
                pct = int(current / total * 100) if total else 0
                cls._print_progress(prefix, f'Descomprimiendo {pct}%')
                stop_event.wait(0.5)

        t = threading.Thread(target=_progress, daemon=True)
        t.start()

        try:
            if cls._find_tool('unzip'):
                subprocess.run(
                    ['unzip', '-q', zip_path, '-d', dest_path],
                    check=True
                )
            else:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dest_path)
        finally:
            stop_event.set()
            t.join()

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
            prefix = f'[{i}/{len(missing_files)}] {filename}'
            verbose = not filename.endswith(('.json', '.md'))
            t0 = time.time()
            cls._download_file(url, local_file, verbose=verbose, prefix=prefix)
            t_download = time.time() - t0

            if filename.endswith('.zip'):
                data_path = os.path.join(local_path, 'data')
                t1 = time.time()
                cls._extract_zip(local_file, data_path, prefix=prefix)
                t_extract = time.time() - t1
                print(f'{prefix}  ✓ ({t_download:.0f}s descarga | {t_extract:.0f}s descompresión){"":<20}')
            elif verbose:
                print(f'{prefix}  ✓ ({t_download:.1f}s){"":<40}')

        open(sentinel, 'w').close()
        return local_path

    # ------------------------------------------------------------------ #
    #  Utilidades de detección                                             #
    # ------------------------------------------------------------------ #

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
    def _find_data_path(cls, local_path):
        """Devuelve la carpeta raíz donde están las subcarpetas de clases."""
        data_path = os.path.join(local_path, 'data')
        return data_path if os.path.exists(data_path) else local_path

    @classmethod
    def _scan_classes(cls, root_path, extensions):
        """
        Escanea root_path buscando subcarpetas como clases.
        Devuelve:
            class_names : list[str]  — nombres de clase ordenados alfabéticamente
            file_list   : list[tuple(path, label_index)]
        """
        subdirs = sorted([
            d for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
        ])
        if not subdirs:
            raise FileNotFoundError(
                f"No se encontraron subcarpetas de clases en {root_path}"
            )

        file_list = []
        for label, subdir in enumerate(subdirs):
            subdir_path = os.path.join(root_path, subdir)
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().endswith(extensions):
                    file_list.append((os.path.join(subdir_path, f), label))

        return subdirs, file_list

    # ------------------------------------------------------------------ #
    #  API pública — tabular                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def list_datasets(cls):
        if cls._repo_files is None:
            cls._repo_files = cls._list_files(filetype=['dir'])
        return cls._repo_files

    @classmethod
    def dataset_path(cls, name):
        return cls._require_repo_directory(name)

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
            raise FileNotFoundError(
                f"No se encontró información sobre el dataset \"{name}\""
            )

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
        return df.columns, df.to_numpy()

    # ------------------------------------------------------------------ #
    #  API pública — archivos (imágenes, audio, genérico)                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_files(cls, name, extensions, loader_fn):
        """
        Carga genérica de archivos organizados en subcarpetas por clase.

        Parámetros
        ----------
        name       : nombre del dataset en el repositorio
        extensions : tupla de extensiones a incluir, ej: ('.wav', '.mp3')
        loader_fn  : función (file_path: str) -> np.ndarray
                     Se llama una vez por archivo y debe devolver un array numpy.

        Retorna
        -------
        X           : np.ndarray — array con todos los samples
        y           : np.ndarray (int) — índices de clase para cada sample
        class_names : list[str] — class_names[y[i]] es la clase del sample i
        """
        local_path = cls._require_repo_directory(name)
        root_path = cls._find_data_path(local_path)
        class_names, file_list = cls._scan_classes(root_path, extensions)

        samples, labels = [], []
        errors = 0
        for file_path, label in tqdm(file_list, desc=f"Cargando {name}",
                                     unit=" archivo", mininterval=2.0):
            try:
                samples.append(loader_fn(file_path))
                labels.append(label)
            except Exception as e:
                errors += 1
                print(f"  [error] {os.path.basename(file_path)}: {e}")

        if errors:
            print(f"  {errors} archivo(s) no pudieron cargarse.")

        return np.array(samples), np.array(labels), class_names

    @classmethod
    def load_files_dataset(cls, name, extensions, loader_fn):
        """
        Versión lazy de load_files. Devuelve un tf.data.Dataset sin configurar.
        El usuario aplica .shuffle(), .batch(), .map() según necesite.

        Parámetros
        ----------
        name       : nombre del dataset en el repositorio
        extensions : tupla de extensiones, ej: ('.png', '.jpg')
        loader_fn  : función (file_path: str) -> np.ndarray

        Retorna
        -------
        ds          : tf.data.Dataset que emite pares (sample, label)
        class_names : list[str]

        Ejemplo
        -------
        ds, class_names = dl.load_files_dataset('fingers', DataLoader.IMAGE_EXTENSIONS,
                                                 mi_loader)
        ds = ds.shuffle(1000).batch(32).map(augmentation)
        model.fit(ds)
        """
        import tensorflow as tf

        local_path = cls._require_repo_directory(name)
        root_path = cls._find_data_path(local_path)
        class_names, file_list = cls._scan_classes(root_path, extensions)

        paths = [fp for fp, _ in file_list]
        labels = [lb for _, lb in file_list]

        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(file_path, label):
            sample = tf.numpy_function(
                func=lambda p: loader_fn(p.decode('utf-8')),
                inp=[file_path],
                Tout=tf.float32
            )
            return sample, label

        ds = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds, class_names

    # ------------------------------------------------------------------ #
    #  Shortcuts para imágenes                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def _default_image_loader(cls, resize=None):
        """Devuelve una loader_fn para imágenes con resize opcional."""
        def loader(file_path):
            img = Image.open(file_path)
            if resize:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float32)
        return loader

    @classmethod
    def load_images(cls, name, resize=None):
        """
        Carga imágenes en memoria como arrays numpy.

        Retorna
        -------
        X           : np.ndarray (N, H, W, C)
        y           : np.ndarray (N,)  — índices de clase
        class_names : list[str]
        """
        return cls.load_files(name, cls.IMAGE_EXTENSIONS,
                               cls._default_image_loader(resize))

    @classmethod
    def load_images_dataset(cls, name, resize=None):
        """
        Versión lazy de load_images. Devuelve tf.data.Dataset sin configurar.

        Ejemplo
        -------
        ds, class_names = dl.load_images_dataset('fingers', resize=(64, 64))
        ds = ds.shuffle(1000).batch(32).map(my_augmentation)
        model.fit(ds)
        """
        return cls.load_files_dataset(name, cls.IMAGE_EXTENSIONS,
                                       cls._default_image_loader(resize))