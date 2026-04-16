#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/30 16:34
@Author  : thezehui@gmail.com
@File    : vector_database_service.py
"""
import logging
import os
from typing import Optional

import weaviate
from injector import inject
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_weaviate import WeaviateVectorStore
from weaviate import WeaviateClient
from weaviate.collections import Collection

from .embeddings_service import EmbeddingsService

# 向量数据库的集合名字
COLLECTION_NAME = "Dataset"


@inject
class VectorDatabaseService:
    """向量数据库服务"""

    client: Optional[WeaviateClient]
    vector_store: Optional[WeaviateVectorStore]
    embeddings_service: EmbeddingsService

    def __init__(self, embeddings_services: EmbeddingsService):
        """构造函数，完成向量数据库服务的客户端+LangChain向量数据库实例的创建"""
        self.embeddings_service = embeddings_services
        self.client = None
        self.vector_store = None
        host = os.getenv("WEAVIATE_HOST")
        port_raw = os.getenv("WEAVIATE_PORT")
        try:
            port = int(port_raw) if port_raw else 8080
            self.client = weaviate.connect_to_local(host=host, port=port)
            self.vector_store = WeaviateVectorStore(
                client=self.client,
                index_name=COLLECTION_NAME,
                text_key="text",
                embedding=self.embeddings_service.cache_backed_embeddings,
            )
        except Exception as exc:
            logging.warning(
                "无法连接 Weaviate（%s:%s），向量检索/索引相关接口将不可用：%s",
                host,
                port_raw,
                exc,
            )

    def get_retriever(self) -> VectorStoreRetriever:
        """获取检索器"""
        if self.vector_store is None:
            raise RuntimeError("Weaviate 未连接，无法创建向量检索器")
        return self.vector_store.as_retriever()

    @property
    def collection(self) -> Collection:
        """当前向量集合；Weaviate 未启动时访问将抛出异常。"""
        if self.client is None:
            raise RuntimeError("Weaviate 未连接，无法访问向量集合")
        return self.client.collections.get(COLLECTION_NAME)
