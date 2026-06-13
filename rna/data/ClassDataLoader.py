import os
import json
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from chardet import UniversalDetector
from PIL import Image
from tqdm import tqdm
import librosa
import tensorflow as tf
import sys
import threading


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
    META_EXTENSIONS = ('.json')

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
    def _initialize_class(cls):
        if cls._resource_path is None:
            cls._base_path = os.path.join(os.getcwd(), '..', 'rna_downloads')
            cls._create_directories()

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
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        if verbose:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=prefix, leave=False, file=sys.stdout) as bar:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                        bar.update(len(chunk))
        else:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)

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
    def _extract_zip(cls, zip_path, dest_path):
        """
        Extrae zip_path en dest_path usando el descompresor más rápido disponible.
        Orden de preferencia: unzip (Linux/Mac) -> zipfile Python (fallback universal).
        Muestra progreso con tqdm actualizado desde un hilo secundario.
        """
        os.makedirs(dest_path, exist_ok=True)
        filename = os.path.basename(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())

        stop_event = threading.Event()

        with tqdm(total=total, unit=' arch', desc=f'  Descomprimiendo {filename}',
                  leave=True, file=sys.stdout) as bar:
            last_count = [0]

            def _progress():
                while not stop_event.is_set():
                    current = sum(len(files) for _, _, files in os.walk(dest_path))
                    delta = current - last_count[0]
                    if delta > 0:
                        bar.update(delta)
                        last_count[0] = current
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
                bar.update(total - last_count[0])

        os.remove(zip_path)

    @classmethod
    def _require_repo_directory(cls, name, force=False):
        name = name.lower()
        local_path = os.path.abspath(os.path.join(cls._resource_path, name))
        os.makedirs(local_path, exist_ok=True)

        sentinel = os.path.join(local_path, '.complete')
        if not force and os.path.exists(sentinel):
            return local_path

        expected_files = cls._list_files(subfolder=name, filetype=['file'])
        if not expected_files:
            raise FileNotFoundError(f"No se encontró el dataset \"{name}\" en el repositorio")

        meta_files = [f for f in expected_files if f.endswith(cls.META_EXTENSIONS)]
        data_files = [f for f in expected_files if not f.endswith(cls.META_EXTENSIONS)]

        # metadata: descargar silencioso sin conteo
        for filename in meta_files:
            local_file = os.path.join(local_path, filename)
            if force or not os.path.exists(local_file):
                url = f"{cls._raw_base_url}/{name}/{filename}"
                cls._download_file(url, local_file, verbose=False)

        # datos: mostrar progreso con conteo
        for i, filename in enumerate(data_files, start=1):
            local_file = os.path.join(local_path, filename)
            prefix = f'[{i}/{len(data_files)}] {filename}'

            if not force and os.path.exists(local_file):
                print(f'  {prefix}  ✓')
                continue

            url = f"{cls._raw_base_url}/{name}/{filename}"
            t0 = time.time()
            cls._download_file(url, local_file, verbose=True, prefix=prefix)
            t_download = time.time() - t0

            if filename.endswith('.zip'):
                data_path = os.path.join(local_path, 'data')
                t1 = time.time()
                cls._extract_zip(local_file, data_path)
                t_extract = time.time() - t1
                print(f'  {prefix}  ✓ ({t_download:.0f}s descarga | {t_extract:.0f}s descompresión)')
            else:
                print(f'  {prefix}  ✓ ({t_download:.1f}s)')

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
                                     unit=" archivo", mininterval=2.0, file=sys.stdout):
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
    def load_files_dataset(
            cls,
            name,
            extensions,
            loader_fn,
            sample_shape=None,
            sample_dtype=tf.float32,
            shuffle=False,
            random_state=None
    ):
        """
        Versión lazy de load_files. Devuelve un tf.data.Dataset sin configurar.

        Parámetros
        ----------
        name          : nombre del dataset en el repositorio
        extensions    : tupla de extensiones válidas
        loader_fn     : función (file_path:str) -> np.ndarray
        shuffle       : mezcla aleatoriamente las muestras
        random_state  : semilla para reproducibilidad

        Retorna
        -------
        ds          : tf.data.Dataset que emite pares (sample, label)
        class_names : list[str]
        """

        local_path = cls._require_repo_directory(name)
        root_path = cls._find_data_path(local_path)
        class_names, file_list = cls._scan_classes(root_path, extensions)

        paths = [fp for fp, _ in file_list]
        labels = [lb for _, lb in file_list]

        # Mezcla previa de rutas y etiquetas
        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(len(paths))

            paths = [paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(file_path, label):
            sample = tf.numpy_function(
                func=lambda p: loader_fn(p.decode('utf-8')),
                inp=[file_path],
                Tout=sample_dtype
            )

            if sample_shape is not None:
                sample.set_shape(sample_shape)

            return sample, label

        ds = path_ds.map(
            _load,
            num_parallel_calls=tf.data.AUTOTUNE
        )

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

            img = np.array(img, dtype=np.float32)

            # Si es gris: (H,W) -> (H,W,1)
            if img.ndim == 2:
                img = img[..., np.newaxis]

            return img

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
    def load_images_dataset(
            cls,
            name,
            resize=None,
            shuffle=False,
            random_state=None
    ):
        """
        Versión lazy de load_images. Devuelve tf.data.Dataset sin configurar.

        Parámetros
        ----------
        name          : nombre del dataset
        resize        : tamaño (ancho, alto)
        shuffle       : mezcla aleatoriamente las muestras
        random_state  : semilla para reproducibilidad

        Retorna
        -------
        ds          : tf.data.Dataset
        class_names : list[str]
        """
        shape = None

        if resize is not None:
            shape = (resize[1], resize[0], 1)

        return cls.load_files_dataset(
            name,
            cls.IMAGE_EXTENSIONS,
            cls._default_image_loader(resize),
            sample_shape=shape,
            shuffle=shuffle,
            random_state=random_state
        )

    @classmethod
    def _default_audio_loader(cls,
                              sample_rate=16000,
                              duration=None,
                              mono=True):

        def loader(file_path):
            audio, sr = librosa.load(
                file_path,
                sr=sample_rate,
                mono=mono
            )

            if duration is not None:
                n_samples = int(sample_rate * duration)

                if len(audio) < n_samples:
                    audio = np.pad(
                        audio,
                        (0, n_samples - len(audio))
                    )
                else:
                    audio = audio[:n_samples]

            return audio.astype(np.float32)

        return loader

    @classmethod
    def load_audio_dataset(cls,
                           name,
                           sample_rate=16000,
                           duration=None):

        sample_shape = None

        if duration is not None:
            sample_shape = (int(sample_rate * duration),)

        return cls.load_files_dataset(
            name,
            cls.AUDIO_EXTENSIONS,
            cls._default_audio_loader(
                sample_rate=sample_rate,
                duration=duration
            ),
            sample_shape=sample_shape
        )


DataLoader._initialize_class()