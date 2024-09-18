import os.path
from os import PathLike
from pathlib import Path
from typing import Optional, List, Union
from urllib.error import HTTPError

import urllib.request

from .llama_embedder import LlamaEmbedder, PoolingType, NormalizationType

__all__ = ["LlamaEmbedder", "PoolingType", "NormalizationType", "Embedder"]

default_cache_dir = Path.home() / ".cache" / "llama_cache"

model_cache_dir = default_cache_dir / "models"


class Embedder(object):
    def __init__(self, model_path: str,
                 *,
                 pooling_type: PoolingType = PoolingType.MEAN,
                 normalization_type: NormalizationType = NormalizationType.EUCLIDEAN,
                 hf_repository: Optional[str] = None,
                 hf_token: Optional[str] = os.environ.get("HF_TOKEN", None)):
        """
        Initializes the embedder

        :param model_path: Path to the model or the model file in the Hugging Face repository (if hf_repository is set)
        :param pooling_type: Pooling type to use. Defaults to PoolingType.MEAN
        :param normalization_type: Normalization type to use for embeddings - defaults to NormalizationType.EUCLIDEAN
        :param hf_repository: Hugging Face repository to download the model from. If set, model_path is the model file within the repository.
        :param hf_token: Hugging Face token to use for downloading the model. Defaults to the HF_TOKEN environment variable.
        """
        if Path(model_path).exists() and Path(model_path).is_dir() and hf_repository is None:
            raise ValueError("If model_path is a directory, hf_repository must be specified")
        self._model_path = model_path
        if hf_repository is not None:
            self._model_path = str(self._download_model(model_path, hf_repository))
        self._pooling_type = pooling_type
        self._normalization_type = normalization_type
        self.embedder = LlamaEmbedder(self._model_path, pooling_type=pooling_type)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts
        :param texts: List of texts to embed
        :return: List of embeddings
        """
        return self.embedder.embed(texts, norm=self._normalization_type)

    @staticmethod
    def _download_model(model_path: str, hf_repository: str, hf_token: Optional[str] = None) -> Union[str, PathLike]:
        """
        Downloads the model from the Hugging Face repository

        :param model_path: Path to the model file in the repository
        :param hf_repository: Hugging Face repository to download the model from
        :param hf_token: Hugging Face token to use for downloading the model
        :return: Path to the downloaded model file
        """

        url = f"https://huggingface.co/{hf_repository}/resolve/main/{model_path}"
        target_file = model_cache_dir / os.path.basename(model_path)
        headers = dict()
        if hf_token is not None:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, headers=headers or {}, method='GET')
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                raise HTTPError(f"Fetching the model failed: {response.status} - {response.reason}")
            with open(target_file, 'wb') as out_file:
                while True:
                    chunk = response.read(1024)
                    if not chunk:
                        break
                    out_file.write(chunk)

        return target_file
