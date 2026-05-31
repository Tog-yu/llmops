#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/8/12 10:50
@Author  : thezehui@gmail.com
@File    : cos_service.py
"""
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flask import current_app, send_from_directory
from injector import inject
from werkzeug.datastructures import FileStorage

from internal.entity.upload_file_entity import ALLOWED_IMAGE_EXTENSION, ALLOWED_DOCUMENT_EXTENSION
from internal.exception import FailException
from internal.model import UploadFile, Account
from .upload_file_service import UploadFileService

# 本地存储根目录
UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent / "storage" / "uploads"


@inject
@dataclass
class CosService:
    """文件存储服务（本地存储模式）"""
    upload_file_service: UploadFileService

    def upload_file(self, file: FileStorage, only_image: bool, account: Account) -> UploadFile:
        """上传文件到本地存储，上传后返回文件的信息"""
        # 1.提取文件扩展名并检测是否可以上传
        filename = file.filename
        extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
        if extension.lower() not in (ALLOWED_IMAGE_EXTENSION + ALLOWED_DOCUMENT_EXTENSION):
            raise FailException(f"该.{extension}扩展的文件不允许上传")
        elif only_image and extension.lower() not in ALLOWED_IMAGE_EXTENSION:
            raise FailException(f"该.{extension}扩展的文件不支持上传，请上传正确的图片")

        # 2.生成一个随机的名字
        random_filename = str(uuid.uuid4()) + "." + extension
        now = datetime.now()
        upload_filename = f"{now.year}/{now.month:02d}/{now.day:02d}/{random_filename}"

        # 3.读取文件内容并保存到本地
        file_content = file.stream.read()
        target_path = UPLOAD_ROOT / upload_filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            target_path.write_bytes(file_content)
        except Exception as e:
            raise FailException("上传文件失败，请稍后重试")

        # 4.创建upload_file记录
        return self.upload_file_service.create_upload_file(
            account_id=account.id,
            name=filename,
            key=upload_filename,
            size=len(file_content),
            extension=extension,
            mime_type=file.mimetype,
            hash=hashlib.sha3_256(file_content).hexdigest(),
        )

    def download_file(self, key: str, target_file_path: str):
        """从本地存储复制文件到指定路径"""
        source = UPLOAD_ROOT / key
        if not source.exists():
            raise FailException("文件不存在")
        shutil.copy2(str(source), target_file_path)

    @classmethod
    def get_file_url(cls, key: str) -> str:
        """根据key生成本地文件访问URL"""
        from flask import request
        return f"{request.scheme}://{request.host}/local-storage/{key}"

    @classmethod
    def serve_local_file(cls, filepath: str):
        """提供本地文件的HTTP访问"""
        file_path = UPLOAD_ROOT / filepath
        if not file_path.exists():
            raise FailException("文件不存在")
        return send_from_directory(str(file_path.parent), file_path.name)
