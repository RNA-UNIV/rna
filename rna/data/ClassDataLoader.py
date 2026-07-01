import os
import json
import subprocess
import time
import unicodedata
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
    """
    Clase singleton para descarga y carga de datasets del repositorio RNA-UNIV.

    Soporta datasets tabulares (CSV), de imágenes y de audio organizados en
    subcarpetas por clase. Los archivos se descargan automáticamente la primera
    vez y se cachean localmente en ``../rna_downloads/data/``.

    Los datasets pueden referenciarse por su nombre canónico o por cualquier
    alias definido en ``dataset_aliases.json`` (ej: ``'wine'`` == ``'vinos'``).
    La resolución de alias es best-effort: si un nombre no está en el índice
    de alias, se asume que ya es el nombre canónico y se intenta usar tal cual.

    Uso básico
    ----------
    ::

        # Tabular
        df, meta = DataLoader.load_dataframe('sonar')

        # Imágenes
        X, y, clases, meta = DataLoader.load_images('natural_scenes_train', resize=(64, 64))

        # Audio
        X, y, clases, meta = DataLoader.load_audio('mi_dataset', fixed_duration=1.0)

        # Lazy (tf.data)
        ds, clases, meta = DataLoader.load_audio_dataset('mi_dataset', fixed_duration=1.0)
    """

    _resource_dir = 'data'
    _repo_files = None
    _instance = None
    _base_path = None
    _base_url = "https://api.github.com/repos/RNA-UNIV/datasets/contents"
    _raw_base_url = "https://raw.githubusercontent.com/RNA-UNIV/datasets/main"
    _releases_base_url = "https://github.com/RNA-UNIV/datasets/releases/download/large_datasets"
    _resource_path = None

    # Alias de datasets (ej: 'vinos' -> 'wine')
    _aliases_file = 'dataset_aliases.json'
    _alias_map = None

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
        """
        Descarga un archivo desde ``url`` a ``local_path``.

        Parámetros
        ----------
        url        : URL del archivo a descargar
        local_path : ruta local donde guardar el archivo
        verbose    : si True muestra barra de progreso tqdm
        prefix     : etiqueta mostrada en la barra de progreso
        """
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
        """
        Verifica si un ejecutable está disponible en el sistema.

        Usa ``which`` en Linux/Mac y un fallback directo en Windows.
        Evita problemas de PATH reducido en entornos como Google Colab.
        """
        try:
            result = subprocess.run(
                ['which', name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            try:
                subprocess.run([name, '--help'], capture_output=True)
                return True
            except FileNotFoundError:
                return False

    @classmethod
    def _extract_zip(cls, zip_path, dest_path):
        """
        Extrae ``zip_path`` en ``dest_path`` usando el descompresor más rápido
        disponible.

        Orden de preferencia: ``unzip`` (Linux/Mac) → ``zipfile`` Python
        (fallback universal). Muestra progreso con tqdm actualizado desde un
        hilo secundario. El archivo zip se elimina tras la extracción.
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

    # ------------------------------------------------------------------ #
    #  Alias de datasets                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(name):
        """
        Normaliza un nombre para comparación: minúsculas, sin acentos,
        espacios/guiones convertidos a ``_``.
        """
        name = name.strip().lower().replace(' ', '_').replace('-', '_')
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        return name

    @classmethod
    def _load_alias_map(cls):
        """
        Carga (y cachea en memoria) el mapa de alias -> nombre canónico.

        Se descarga una única vez ``dataset_aliases.json`` desde el
        repositorio y se cachea localmente. Es un mecanismo best-effort:
        si el archivo no existe o falla la descarga, se continúa sin
        alias (no rompe el acceso a datasets por su nombre canónico).
        """
        if cls._alias_map is not None:
            return cls._alias_map

        local_file = os.path.join(cls._resource_path, cls._aliases_file)
        if not os.path.exists(local_file):
            try:
                url = f"{cls._raw_base_url}/{cls._aliases_file}"
                cls._download_file(url, local_file, verbose=False)
            except requests.RequestException:
                cls._alias_map = {}
                return cls._alias_map

        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            cls._alias_map = {}
            return cls._alias_map

        alias_map = {}
        for canonical, alias_list in raw.items():
            alias_map[cls._normalize(canonical)] = canonical
            for alias in alias_list:
                alias_map[cls._normalize(alias)] = canonical

        cls._alias_map = alias_map
        return cls._alias_map

    @classmethod
    def _resolve(cls, name):
        """
        Resuelve un posible alias al nombre canónico del dataset.

        Si ``name`` no aparece en el índice de alias, se devuelve tal
        cual: se asume que ya es el nombre canónico y es el flujo normal
        de descarga (``_require_repo_directory``) el que determina si
        el dataset existe o no.
        """
        alias_map = cls._load_alias_map()
        key = cls._normalize(name)
        return alias_map.get(key, name)

    @classmethod
    def _require_repo_directory(cls, name, force=False):
        """
        Garantiza que el dataset ``name`` esté disponible localmente.

        Resuelve alias antes de cualquier otra cosa. Intenta descargar
        primero desde el repositorio raw y luego desde GitHub Releases
        para archivos grandes. Usa un archivo centinela ``.complete``
        para evitar descargas repetidas.

        Parámetros
        ----------
        name  : nombre del dataset (se resuelve el alias y se normaliza
                a minúsculas)
        force : si True fuerza la descarga aunque el dataset ya exista

        Retorna
        -------
        local_path : ruta absoluta al directorio del dataset

        Raises
        ------
        FileNotFoundError : si el dataset no se encuentra en el repositorio
        """
        name = cls._resolve(name).lower()
        local_path = os.path.abspath(os.path.join(cls._resource_path, name))
        os.makedirs(local_path, exist_ok=True)

        sentinel = os.path.join(local_path, '.complete')
        if not force and os.path.exists(sentinel):
            return local_path

        for filename in [f'{name}.csv', f'{name}.zip']:
            local_file = os.path.join(local_path, filename)
            url_raw = f"{cls._raw_base_url}/{name}/{filename}"
            url_release = f"{cls._releases_base_url}/{filename}"

            try:
                cls._download_file(url_raw, local_file, verbose=True, prefix=filename)
            except requests.HTTPError:
                try:
                    cls._download_file(url_release, local_file, verbose=True, prefix=filename)
                except requests.HTTPError:
                    continue

            if filename.endswith('.zip'):
                with zipfile.ZipFile(local_file, 'r') as zf:
                    has_csv = any(n.endswith('.csv') for n in zf.namelist())
                dest_path = local_path if has_csv else os.path.join(local_path, 'data')
                t1 = time.time()
                cls._extract_zip(local_file, dest_path)
                t_extract = time.time() - t1
                print(f'  {filename}  ✓ ({t_extract:.0f}s descompresión)')
            else:
                print(f'  {filename}  ✓')

            open(sentinel, 'w').close()
            return local_path

        raise FileNotFoundError(f"No se encontró el dataset \"{name}\" en el repositorio")

    # ------------------------------------------------------------------ #
    #  Utilidades de detección                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def _detect_encoding(cls, file_path):
        """Detecta la codificación de un archivo de texto usando chardet."""
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
        """
        Detecta el separador de un CSV inspeccionando la primera línea.

        Evalúa ``,``, ``;`` y ``\\t`` y devuelve el que aparece más veces.
        """
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
            separators = [',', ';', '\t']
            return max(separators, key=lambda sep: first_line.count(sep))

    @classmethod
    def _find_data_path(cls, local_path):
        """
        Devuelve la carpeta raíz donde están las subcarpetas de clases.

        Si existe una subcarpeta ``data/`` dentro de ``local_path``, la
        devuelve; de lo contrario devuelve ``local_path`` directamente.
        """
        data_path = os.path.join(local_path, 'data')
        return data_path if os.path.exists(data_path) else local_path

    @classmethod
    def _scan_classes(cls, root_path, extensions):
        """
        Escanea ``root_path`` buscando subcarpetas como clases.

        Cada subcarpeta directa de ``root_path`` se trata como una clase.
        Dentro de cada una se listan los archivos cuya extensión coincida
        con ``extensions``.

        Parámetros
        ----------
        root_path  : directorio raíz con subcarpetas de clases
        extensions : tupla de extensiones válidas, ej: ``('.wav', '.mp3')``

        Retorna
        -------
        class_names : list[str]  — nombres de clase ordenados alfabéticamente
        file_list   : list[tuple(str, int)]  — pares (ruta_archivo, índice_clase)

        Raises
        ------
        FileNotFoundError : si no se encuentran subcarpetas en ``root_path``
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
        """
        Lista los datasets disponibles en el repositorio RNA-UNIV.

        El resultado se cachea en memoria tras la primera consulta.

        Retorna
        -------
        list[str] — nombres de los datasets disponibles
        """
        if cls._repo_files is None:
            cls._repo_files = cls._list_files(filetype=['dir'])
        return cls._repo_files

    @classmethod
    def dataset_path(cls, name):
        """
        Devuelve la ruta local al dataset, descargándolo si es necesario.

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name : nombre del dataset (o alias)

        Retorna
        -------
        str — ruta absoluta al directorio local del dataset
        """
        return cls._require_repo_directory(name)

    @classmethod
    def dataset_info(cls, name, force=False):
        """
        Descarga y retorna el archivo ``info.json`` del dataset.

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name  : nombre del dataset (o alias)
        force : si True fuerza la descarga aunque el archivo ya exista

        Retorna
        -------
        dict — contenido del ``info.json``

        Raises
        ------
        FileNotFoundError : si no existe ``info.json`` para el dataset
        """
        name = cls._resolve(name).lower()
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
        """
        Muestra el ``info.json`` del dataset de forma legible.

        En entornos Jupyter/Colab usa ``IPython.display.JSON`` para
        renderizado interactivo; en otros entornos imprime JSON indentado.

        Parámetros
        ----------
        name  : nombre del dataset (o alias)
        force : si True fuerza la descarga del ``info.json``
        """
        info_data = cls.dataset_info(name, force)
        try:
            from IPython import get_ipython
            from IPython.display import display, JSON
            if get_ipython() is not None:
                display(JSON(info_data, root=cls._resolve(name)))
            else:
                raise RuntimeError
        except (ImportError, RuntimeError):
            print(json.dumps(info_data, indent=2, ensure_ascii=False))

    @classmethod
    def load_dataframe(cls, name, encoding=None, separator=None, return_metadata=False):
        """
        Carga un dataset tabular como ``pd.DataFrame``.

        Acepta nombre canónico o alias. La codificación y el separador se
        detectan automáticamente si no se especifican. Los valores típicos
        de missing (``'?'``, ``'NA'``, ``'null'``, etc.) se reemplazan por
        ``NaN``.

        Parámetros
        ----------
        name            : nombre del dataset (o alias) en el repositorio
        encoding        : codificación del CSV (ej: ``'utf-8'``); se autodetecta si es None
        separator       : separador del CSV (``','``, ``';'``, ``'\\t'``); se autodetecta si es None
        return_metadata : si True retorna también un dict con ``'encoding'`` y
                          ``'separator'`` efectivamente usados (default: False)

        Retorna
        -------
        df                    : pd.DataFrame
        metadata (opcional)   : dict con claves ``'encoding'`` y ``'separator'``
        """
        local_path = cls._require_repo_directory(name)
        csv_files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No se encontró archivo CSV en {local_path}")

        file_path = os.path.join(local_path, csv_files[0])
        if encoding is None:
            encoding = cls._detect_encoding(file_path)
        if separator is None:
            separator = cls._detect_separator(file_path, encoding)

        df = pd.read_csv(file_path,
                         na_values=['?', ' ', 'NA', 'N/A', 'null', '-', 'unknown', ''],
                         encoding=encoding,
                         sep=separator)

        if return_metadata:
            return df, {'encoding': encoding, 'separator': separator}
        return df

    @classmethod
    def load_array(cls, name, encoding=None, separator=None, return_type='mixed',
                   return_metadata=False):
        """
        Carga un dataset tabular como array numpy.

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name            : nombre del dataset (o alias)
        encoding        : codificación del CSV; se autodetecta si es None
        separator       : separador del CSV; se autodetecta si es None
        return_type     : controla qué columnas se incluyen:

                          - ``'mixed'``       — todo como ``object`` (default)
                          - ``'numeric'``     — solo columnas numéricas
                          - ``'categorical'`` — solo columnas categóricas
                          - ``'both'``        — dict con arrays numérico y categórico

        return_metadata : si True agrega un dict con ``'encoding'`` y
                          ``'separator'`` al final del retorno (default: False)

        Retorna
        -------
        Si ``return_type`` es ``'mixed'``, ``'numeric'`` o ``'categorical'``:
            columns             : pd.Index — nombres de columnas
            data                : np.ndarray
            metadata (opcional) : dict con claves ``'encoding'`` y ``'separator'``

        Si ``return_type`` es ``'both'``:
            result              : dict con claves ``'numeric_columns'``,
                                  ``'numeric_data'``, ``'categorical_columns'``,
                                  ``'categorical_data'``
            metadata (opcional) : dict con claves ``'encoding'`` y ``'separator'``

        Raises
        ------
        ValueError : si ``return_type`` no es uno de los valores válidos
        """
        df, metadata = cls.load_dataframe(name, encoding, separator, return_metadata=True)

        if return_type == 'mixed':
            result = (df.columns, df.to_numpy())

        elif return_type == 'numeric':
            numeric_df = df.select_dtypes(include=[np.number])
            result = (numeric_df.columns, numeric_df.to_numpy())

        elif return_type == 'categorical':
            cat_df = df.select_dtypes(include=['object', 'category'])
            result = (cat_df.columns, cat_df.to_numpy())

        elif return_type == 'both':
            numeric_df = df.select_dtypes(include=[np.number])
            cat_df = df.select_dtypes(include=['object', 'category'])
            both_result = {
                'numeric_columns': numeric_df.columns,
                'numeric_data': numeric_df.to_numpy(),
                'categorical_columns': cat_df.columns,
                'categorical_data': cat_df.to_numpy()
            }
            if return_metadata:
                return both_result, metadata
            return both_result

        else:
            raise ValueError(
                f"return_type debe ser 'mixed', 'numeric', 'categorical' o 'both', "
                f"se recibió: '{return_type}'"
            )

        # return_type in ('mixed', 'numeric', 'categorical')
        if return_metadata:
            return (*result, metadata)
        return result

    # ------------------------------------------------------------------ #
    #  API pública — archivos (imágenes, audio, genérico)                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_files(cls, name, extensions, loader_fn):
        """
        Carga genérica de archivos organizados en subcarpetas por clase.

        Acepta nombre canónico o alias. API de bajo nivel: el usuario
        provee su propia función de carga. Para imágenes y audio se
        recomienda usar ``load_images`` y ``load_audio`` respectivamente.

        Parámetros
        ----------
        name       : nombre del dataset (o alias) en el repositorio
        extensions : tupla de extensiones a incluir, ej: ``('.wav', '.mp3')``
        loader_fn  : función ``(file_path: str) -> np.ndarray``
                     Se llama una vez por archivo.

        Retorna
        -------
        X           : np.ndarray — array con todos los samples apilados
        y           : np.ndarray (int) — índices de clase para cada sample
        class_names : list[str] — ``class_names[y[i]]`` es la clase del sample ``i``
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
        Versión lazy de ``load_files``. Devuelve un ``tf.data.Dataset``
        sin configurar (sin batch, prefetch ni augmentation).

        Acepta nombre canónico o alias. API de bajo nivel: para imágenes
        y audio se recomienda usar ``load_images_dataset`` y
        ``load_audio_dataset`` respectivamente.

        Parámetros
        ----------
        name          : nombre del dataset (o alias) en el repositorio
        extensions    : tupla de extensiones válidas
        loader_fn     : función ``(file_path: str) -> np.ndarray``
        sample_shape  : shape estático del sample para ``set_shape``; útil
                        cuando todos los samples tienen la misma forma
        sample_dtype  : dtype TensorFlow de los samples (default: ``tf.float32``)
        shuffle       : si True mezcla aleatoriamente antes de crear el dataset
        random_state  : semilla para reproducibilidad del shuffle

        Retorna
        -------
        ds          : tf.data.Dataset que emite pares ``(sample, label)``
        class_names : list[str]
        """
        local_path = cls._require_repo_directory(name)
        root_path = cls._find_data_path(local_path)
        class_names, file_list = cls._scan_classes(root_path, extensions)

        paths = [fp for fp, _ in file_list]
        labels = [lb for _, lb in file_list]

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

        ds = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

        return ds, class_names

    # ------------------------------------------------------------------ #
    #  Shortcuts para imágenes                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def _default_image_loader(cls, resize=None):
        """
        Devuelve una ``loader_fn`` para imágenes.

        Parámetros
        ----------
        resize : tupla ``(ancho, alto)`` para redimensionar; None conserva
                 el tamaño original

        Retorna
        -------
        loader : función ``(file_path: str) -> np.ndarray`` de shape
                 ``(H, W, C)`` en ``float32``. Imágenes en escala de grises
                 se expanden a ``(H, W, 1)``.
        """
        def loader(file_path):
            img = Image.open(file_path)
            if resize:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            img = np.array(img, dtype=np.float32)
            if img.ndim == 2:
                img = img[..., np.newaxis]
            return img

        return loader

    @classmethod
    def load_images(cls, name, resize=None):
        """
        Carga imágenes en memoria como arrays numpy.

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name   : nombre del dataset (o alias) en el repositorio
        resize : tupla ``(ancho, alto)`` para redimensionar todas las imágenes
                 a un tamaño común. Si es None se conserva el tamaño original
                 (puede fallar al apilar si las imágenes tienen distintos tamaños).

        Retorna
        -------
        X           : np.ndarray ``(N, H, W, C)`` en ``float32``
        y           : np.ndarray ``(N,)`` — índices de clase
        class_names : list[str]
        metadata    : dict con claves:

                      - ``'color_space'``: ``'rgb'`` (valor fijo actual)
                      - ``'resize'``: valor del parámetro ``resize``
        """
        X, y, class_names = cls.load_files(
            name,
            cls.IMAGE_EXTENSIONS,
            cls._default_image_loader(resize)
        )
        metadata = {'color_space': 'rgb', 'resize': resize}
        return X, y, class_names, metadata

    @classmethod
    def load_images_dataset(cls, name, resize=None, shuffle=False, random_state=None):
        """
        Versión lazy de ``load_images``. Devuelve un ``tf.data.Dataset``
        sin configurar (sin batch, prefetch ni augmentation).

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name         : nombre del dataset (o alias)
        resize       : tupla ``(ancho, alto)``; None conserva tamaño original
        shuffle      : si True mezcla aleatoriamente las muestras
        random_state : semilla para reproducibilidad del shuffle

        Retorna
        -------
        ds          : tf.data.Dataset que emite pares ``(imagen, label)``
        class_names : list[str]
        metadata    : dict con claves:

                      - ``'color_space'``: ``'rgb'``
                      - ``'resize'``: valor del parámetro ``resize``
        """
        sample_shape = None
        if resize is not None:
            sample_shape = (resize[1], resize[0], 1)

        ds, class_names = cls.load_files_dataset(
            name,
            cls.IMAGE_EXTENSIONS,
            cls._default_image_loader(resize),
            sample_shape=sample_shape,
            shuffle=shuffle,
            random_state=random_state
        )
        metadata = {'color_space': 'rgb', 'resize': resize}
        return ds, class_names, metadata

    # ------------------------------------------------------------------ #
    #  Shortcuts para audio                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def _default_audio_loader(cls, sample_rate=None, fixed_duration=None, mono=True):
        """
        Devuelve una ``loader_fn`` para archivos de audio.

        Parámetros
        ----------
        sample_rate    : frecuencia de muestreo objetivo en Hz. Si es None
                         se respeta el sample rate original del archivo
                         (equivalente a ``librosa.load(sr=None)``).
        fixed_duration : duración fija en segundos a la que se normaliza cada
                         audio. Los audios más cortos se rellenan con ceros;
                         los más largos se recortan. Si es None se respeta la
                         duración original.
        mono           : si True convierte a mono ``(T,)``; si False conserva
                         los canales en formato ``(C, T)``.

        Retorna
        -------
        loader : función ``(file_path: str) -> np.ndarray`` en ``float32``
        """
        def loader(file_path):
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=mono)

            if fixed_duration is not None:
                # Usar el sr efectivo: el solicitado o el original del archivo
                effective_sr = sample_rate if sample_rate is not None else sr
                n_samples = int(effective_sr * fixed_duration)

                if mono:
                    # shape: (T,)
                    if len(audio) < n_samples:
                        audio = np.pad(audio, (0, n_samples - len(audio)))
                    else:
                        audio = audio[:n_samples]
                else:
                    # shape: (C, T) — librosa devuelve canales en eje 0
                    T = audio.shape[-1]
                    if T < n_samples:
                        audio = np.pad(audio, ((0, 0), (0, n_samples - T)))
                    else:
                        audio = audio[..., :n_samples]

            return audio.astype(np.float32)

        return loader

    @classmethod
    def _detect_sample_rate(cls, name):
        """
        Detecta el sample rate del primer archivo de audio del dataset.

        Se usa ``librosa.get_samplerate`` para evitar cargar el audio completo.

        Parámetros
        ----------
        name : nombre canónico del dataset (ya resuelto; debe estar
               disponible localmente)

        Retorna
        -------
        int — sample rate en Hz, o None si no se encuentra ningún archivo
        """
        local_path = os.path.join(cls._resource_path, name.lower())
        root_path = cls._find_data_path(local_path)

        for dirpath, _, filenames in os.walk(root_path):
            for fname in sorted(filenames):
                if fname.lower().endswith(cls.AUDIO_EXTENSIONS):
                    return librosa.get_samplerate(os.path.join(dirpath, fname))
        return None

    @classmethod
    def load_audio(cls, name, sample_rate=None, fixed_duration=None, mono=True):
        """
        Carga archivos de audio en memoria como arrays numpy.

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name           : nombre del dataset (o alias) en el repositorio
        sample_rate    : frecuencia de muestreo objetivo en Hz. Si es None
                         se respeta el sample rate original de cada archivo.
                         Si los archivos tienen sample rates distintos y no se
                         especifica ``fixed_duration``, el apilado puede fallar.
        fixed_duration : duración fija en segundos. Los audios más cortos se
                         rellenan con ceros; los más largos se recortan.
                         Si es None se respeta la duración original de cada archivo.
        mono           : si True convierte a mono; si False conserva los canales.

        Retorna
        -------
        X           : np.ndarray ``(N, T)`` si mono, ``(N, C, T)`` si no
        y           : np.ndarray ``(N,)`` — índices de clase
        class_names : list[str]
        metadata    : dict con claves:

                      - ``'sample_rate'``: sr efectivamente usado (detectado
                        automáticamente si no se especificó)
                      - ``'mono'``: valor del parámetro ``mono``
                      - ``'fixed_duration'``: valor del parámetro ``fixed_duration``
        """
        # Resolver alias una sola vez: se reutiliza tanto para la descarga
        # como para la detección de sample rate.
        name = cls._resolve(name)

        # Descarga el dataset antes de intentar detectar el sr
        cls._require_repo_directory(name)

        effective_sr = sample_rate
        if effective_sr is None:
            effective_sr = cls._detect_sample_rate(name)
            if effective_sr is not None:
                print(f"  sample_rate detectado: {effective_sr} Hz")

        X, y, class_names = cls.load_files(
            name,
            cls.AUDIO_EXTENSIONS,
            cls._default_audio_loader(
                sample_rate=sample_rate,
                fixed_duration=fixed_duration,
                mono=mono
            )
        )
        metadata = {
            'sample_rate': effective_sr,
            'mono': mono,
            'fixed_duration': fixed_duration
        }
        return X, y, class_names, metadata

    @classmethod
    def load_audio_dataset(cls,
                           name,
                           sample_rate=None,
                           fixed_duration=None,
                           mono=True,
                           shuffle=False,
                           random_state=None):
        """
        Versión lazy de ``load_audio``. Devuelve un ``tf.data.Dataset``
        sin configurar (sin batch, prefetch ni augmentation).

        Acepta nombre canónico o alias.

        Parámetros
        ----------
        name           : nombre del dataset (o alias) en el repositorio
        sample_rate    : frecuencia de muestreo objetivo en Hz. Si es None
                         se respeta el sample rate original de cada archivo.
        fixed_duration : duración fija en segundos para normalizar todos los
                         audios. Si es None el shape del sample no se fija
                         (requiere batch con padding o audios de igual longitud).
        mono           : si True convierte a mono; si False conserva los canales.
        shuffle        : si True mezcla aleatoriamente las muestras
        random_state   : semilla para reproducibilidad del shuffle

        Retorna
        -------
        ds          : tf.data.Dataset que emite pares ``(audio, label)``
        class_names : list[str]
        metadata    : dict con claves:

                      - ``'sample_rate'``: sr efectivamente usado (detectado
                        automáticamente si no se especificó)
                      - ``'mono'``: valor del parámetro ``mono``
                      - ``'fixed_duration'``: valor del parámetro ``fixed_duration``

        Nota
        ----
        A diferencia de ``load_audio``, el dataset es lazy: los archivos no se
        leen al llamar esta función sino al iterar el dataset. Por eso el
        ``sample_rate`` se detecta inspeccionando el primer archivo sin cargarlo
        completo (``librosa.get_samplerate``).
        """
        # Resolver alias una sola vez: se reutiliza tanto para la descarga
        # como para la detección de sample rate.
        name = cls._resolve(name)

        # Descarga el dataset antes de intentar detectar el sr
        cls._require_repo_directory(name)

        effective_sr = sample_rate
        if effective_sr is None:
            effective_sr = cls._detect_sample_rate(name)
            if effective_sr is not None:
                print(f"  sample_rate detectado: {effective_sr} Hz")

        sample_shape = None
        if fixed_duration is not None and effective_sr is not None:
            n = int(effective_sr * fixed_duration)
            sample_shape = (n,) if mono else (2, n)

        ds, class_names = cls.load_files_dataset(
            name,
            cls.AUDIO_EXTENSIONS,
            cls._default_audio_loader(
                sample_rate=sample_rate,
                fixed_duration=fixed_duration,
                mono=mono
            ),
            sample_shape=sample_shape,
            sample_dtype=tf.float32,
            shuffle=shuffle,
            random_state=random_state
        )
        metadata = {
            'sample_rate': effective_sr,
            'mono': mono,
            'fixed_duration': fixed_duration
        }
        return ds, class_names, metadata


DataLoader._initialize_class()