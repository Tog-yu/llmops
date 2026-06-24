#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/8/30 13:35
@Author  : thezehui@gmail.com
@File    : embeddings_service.py
"""
import os
from dataclasses import dataclass

import tiktoken
from injector import inject
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.storage import RedisStore
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from redis import Redis


@inject
@dataclass
class EmbeddingsService:
    """文本嵌入模型服务"""
    _store: RedisStore
    _embeddings: Embeddings
    _cache_backed_embeddings: CacheBackedEmbeddings

    def __init__(self, redis: Redis):
        """构造函数，初始化文本嵌入模型客户端、存储器、缓存客户端"""
        self._store = RedisStore(client=redis)
        cache_folder = os.path.join(os.getcwd(), "internal", "core", "embeddings")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")

        if embedding_provider in {"local", "huggingface"}:
            embedding_provider = "local"
            # Force Hugging Face caches into the repo-local writable directory.
            os.environ.setdefault("HF_HOME", cache_folder)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_folder)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_folder)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                cache_folder=cache_folder,
                model_kwargs={
                    "trust_remote_code": True,
                    "local_files_only": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )
        else:
            if embedding_provider != "ollama":
                embedding_provider = "ollama"
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
            self._embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            )
        self._cache_backed_embeddings = CacheBackedEmbeddings.from_bytes_store(
            self._embeddings,
            self._store,
            namespace=f"embeddings:{embedding_provider}:{embedding_model}",
        )

    @classmethod
    def calculate_token_count(cls, query: str) -> int:
        """计算传入文本的token数"""
        encoding = tiktoken.encoding_for_model("gpt-3.5")
        return len(encoding.encode(query))

    @property
    def store(self) -> RedisStore:
        return self._store

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def cache_backed_embeddings(self) -> CacheBackedEmbeddings:
        return self._cache_backed_embeddings
