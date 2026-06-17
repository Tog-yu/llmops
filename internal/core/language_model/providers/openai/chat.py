#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/01 11:48
@Author  : thezehui@gmail.com
@File    : chat.py
"""
from typing import Tuple

import tiktoken
from langchain_openai import ChatOpenAI

from internal.core.language_model.entities.model_entity import BaseLanguageModel


class Chat(ChatOpenAI, BaseLanguageModel):
    """OpenAI聊天模型基类"""

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        """重写获取编码模型，兼容tiktoken尚未收录的新模型名(如gpt-5.4-mini)"""
        try:
            model, encoding = super()._get_encoding_model()
            # 确保返回的模型名能通过get_num_tokens_from_messages的白名单检测
            if not model.startswith("gpt-3.5-turbo") and not model.startswith("gpt-4"):
                return "gpt-4", encoding
            return model, encoding
        except KeyError:
            return "gpt-4", tiktoken.encoding_for_model("gpt-4")
