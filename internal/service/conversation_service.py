#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/09/28 0:15
@Author  : thezehui@gmail.com
@File    : conversation_service.py
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from flask import Flask
from injector import inject
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import desc

from internal.core.agent.entities.queue_entity import AgentThought, QueueEvent
from internal.entity.conversation_entity import (
    SUMMARIZER_TEMPLATE,
    CONVERSATION_NAME_TEMPLATE,
    ConversationInfo,
    SUGGESTED_QUESTIONS_TEMPLATE,
    SuggestedQuestions, InvokeFrom, MessageStatus,
)
from internal.exception import NotFoundException
from internal.model import Conversation, Message, MessageAgentThought, Account
from internal.schema.conversation_schema import GetConversationMessagesWithPageReq
from pkg.paginator import Paginator
from pkg.sqlalchemy import SQLAlchemy
from .base_service import BaseService


@inject
@dataclass
class ConversationService(BaseService):
    """会话服务"""
    db: SQLAlchemy

    @classmethod
    def summary(cls, human_message: str, ai_message: str, old_summary: str = "") -> str:
        """根据传递的人类消息、AI消息还有原始的摘要信息总结生成一段新的摘要"""
        # 1.创建prompt
        prompt = ChatPromptTemplate.from_template(SUMMARIZER_TEMPLATE)

        # 2.构建大语言模型实例，并且将大语言模型的温度调低，降低幻觉的概率
        llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"), temperature=0.5)

        # 3.构建链应用
        summary_chain = prompt | llm | StrOutputParser()

        # 4.调用链并获取新摘要信息
        new_summary = summary_chain.invoke({
            "summary": old_summary,
            "new_lines": f"Human: {human_message}\nAI: {ai_message}",
        })

        return new_summary

    @classmethod
    def generate_conversation_name(cls, query: str) -> str:
        """根据传递的query生成对应的会话名字，并且语言与用户的输入保持一致"""
        # 1.创建prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONVERSATION_NAME_TEMPLATE),
            ("human", "{query}")
        ])

        # 2.构建大语言模型实例，并且将大语言模型的温度调低，降低幻觉的概率
        llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"), temperature=0)
        structured_llm = llm.with_structured_output(ConversationInfo)

        # 3.构建链应用
        chain = prompt | structured_llm

        # 4.提取并整理query，截取长度过长的部分
        if len(query) > 2000:
            query = query[:300] + "...[TRUNCATED]..." + query[-300:]
        query = query.replace("\n", " ")

        # 5.调用链并获取会话信息
        conversation_info = chain.invoke({"query": query})

        # 6.提取会话名称
        name = "新的会话"
        try:
            if conversation_info and hasattr(conversation_info, "subject"):
                name = conversation_info.subject
        except Exception as e:
            logging.exception(f"提取会话名称出错, conversation_info: {conversation_info}, 错误信息: {str(e)}")
        if len(name) > 75:
            name = name[:75] + "..."

        return name

    @classmethod
    def generate_suggested_questions(cls, histories: str) -> list[str]:
        """根据传递的历史信息生成最多不超过3个的建议问题"""
        # 1.创建prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUGGESTED_QUESTIONS_TEMPLATE),
            ("human", "{histories}")
        ])

        # 2.构建大语言模型实例，并且将大语言模型的温度调低，降低幻觉的概率
        llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"), temperature=0)
        structured_llm = llm.with_structured_output(SuggestedQuestions)

        # 3.构建链应用
        chain = prompt | structured_llm

        # 4.调用链并获取建议问题列表
        suggested_questions = chain.invoke({"histories": histories})

        # 5.提取建议问题列表
        questions = []
        try:
            if suggested_questions and hasattr(suggested_questions, "questions"):
                questions = suggested_questions.questions
        except Exception as e:
            logging.exception(f"生成建议问题出错, suggested_questions: {suggested_questions}, 错误信息: {str(e)}")
        if len(questions) > 3:
            questions = questions[:3]

        return questions

    def save_agent_thoughts(
            self,
            flask_app: Flask,
            account_id: UUID,
            app_id: UUID,
            app_config: dict[str, Any],
            conversation_id: UUID,
            message_id: UUID,
            agent_thoughts: list[AgentThought],
    ):
        """在子线程中异步执行，将Agent流式输出的推理步骤持久化到3张表(message_agent_thought, message, conversation)"""
        # 1.通过传递的flask_app建立应用上下文，使子线程能正常使用db.session等Flask扩展
        with flask_app.app_context():
            # 2.在子线程中重新查询conversation和message(主线程的orm对象不能跨线程使用)
            conversation = self.get(Conversation, conversation_id)
            message = self.get(Message, message_id)

            # 3.定义变量记录推理步骤位置和总耗时
            position = 0
            latency = 0

            # 4.遍历所有推理步骤，依次写入数据库
            for agent_thought in agent_thoughts:
                # ====== 写入 message_agent_thought 表 ======
                # 存储长期记忆召回、推理、消息、动作、知识库检索等Agent推理步骤明细
                if agent_thought.event in [
                    QueueEvent.LONG_TERM_MEMORY_RECALL,
                    QueueEvent.AGENT_THOUGHT,
                    QueueEvent.AGENT_MESSAGE,
                    QueueEvent.AGENT_ACTION,
                    QueueEvent.DATASET_RETRIEVAL,
                ]:
                    position += 1
                    latency += agent_thought.latency

                    self.create(
                        MessageAgentThought,
                        app_id=app_id,
                        conversation_id=conversation.id,
                        message_id=message.id,
                        invoke_from=InvokeFrom.DEBUGGER,
                        created_by=account_id,
                        position=position,
                        event=agent_thought.event,
                        thought=agent_thought.thought,
                        observation=agent_thought.observation,
                        tool=agent_thought.tool,
                        tool_input=agent_thought.tool_input,
                        # 消息相关数据(Prompt)
                        message=agent_thought.message,
                        message_token_count=agent_thought.message_token_count,
                        message_unit_price=agent_thought.message_unit_price,
                        message_price_unit=agent_thought.message_price_unit,
                        # 答案相关数据(LLM生成的回答)
                        answer=agent_thought.answer,
                        answer_token_count=agent_thought.answer_token_count,
                        answer_unit_price=agent_thought.answer_unit_price,
                        answer_price_unit=agent_thought.answer_price_unit,
                        # Agent推理统计相关(token消耗与成本)
                        total_token_count=agent_thought.total_token_count,
                        total_price=agent_thought.total_price,
                        latency=agent_thought.latency,
                    )

                # ====== 写入 message 表 ======
                # AGENT_MESSAGE是LLM最终的文本回答，需要更新message记录的answer和统计信息
                if agent_thought.event == QueueEvent.AGENT_MESSAGE:
                    self.update(
                        message,
                        # 消息相关字段(请求LLM的Prompt消息列表)
                        message=agent_thought.message,
                        message_token_count=agent_thought.message_token_count,
                        message_unit_price=agent_thought.message_unit_price,
                        message_price_unit=agent_thought.message_price_unit,
                        # 答案相关字段(LLM生成的最终回答)
                        answer=agent_thought.answer,
                        answer_token_count=agent_thought.answer_token_count,
                        answer_unit_price=agent_thought.answer_unit_price,
                        answer_price_unit=agent_thought.answer_price_unit,
                        # 统计信息
                        total_token_count=agent_thought.total_token_count,
                        total_price=agent_thought.total_price,
                        latency=latency,  # 使用累计总耗时
                    )

                    # ====== 写入 conversation 表(长期记忆) ======
                    # 如果开启了长期记忆，用LLM增量摘要更新conversation.summary
                    if app_config["long_term_memory"]["enable"]:
                        new_summary = self.summary(
                            message.query,      # 用户本轮问题
                            agent_thought.answer,  # LLM本轮回答
                            conversation.summary   # 旧摘要
                        )
                        self.update(
                            conversation,
                            summary=new_summary,
                        )

                    # ====== 写入 conversation 表(会话名称) ======
                    # 如果是新创建的会话，根据第一轮query生成会话名称
                    if conversation.is_new:
                        new_conversation_name = self.generate_conversation_name(message.query)
                        self.update(
                            conversation,
                            name=new_conversation_name,
                        )

                # ====== 处理异常终止 ======
                # 如果是超时/停止/错误事件，更新message状态并终止后续写入
                if agent_thought.event in [QueueEvent.TIMEOUT, QueueEvent.STOP, QueueEvent.ERROR]:
                    self.update(
                        message,
                        status=agent_thought.event,
                        error=agent_thought.observation,
                    )
                    break

    def get_conversation(self, conversation_id: UUID, account: Account) -> Conversation:
        """根据传递的会话id+account，获取指定的会话信息"""
        # 1.根据conversation_id查询会话记录
        conversation = self.get(Conversation, conversation_id)
        if (
                not conversation
                or conversation.created_by != account.id
                or conversation.is_deleted
        ):
            raise NotFoundException("该会话不存在或被删除，请核实后重试")

        # 2.校验通过返回会话
        return conversation

    def get_message(self, message_id: UUID, account: Account) -> Message:
        """根据传递的消息id+账号，获取指定的消息"""
        # 1.根据message_id查询消息记录
        message = self.get(Message, message_id)
        if (
                not message
                or message.created_by != account.id
                or message.is_deleted
        ):
            raise NotFoundException("该消息不存在或被删除，请核实后重试")

        # 2.校验通过返回消息
        return message

    def get_conversation_messages_with_page(
            self,
            conversation_id: UUID,
            req: GetConversationMessagesWithPageReq,
            account: Account,
    ) -> tuple[list[Message], Paginator]:
        """根据传递的会话id+请求数据，获取当前账号下该会话的消息分页列表数据"""
        # 1.获取会话并校验权限
        conversation = self.get_conversation(conversation_id, account)

        # 2.构建分页器并设置游标条件
        paginator = Paginator(db=self.db, req=req)
        filters = []
        if req.created_at.data:
            # 3.将时间戳转换成DateTime
            created_at_datetime = datetime.fromtimestamp(req.created_at.data)
            filters.append(Message.created_at <= created_at_datetime)

        # 4.执行分页并查询数据
        messages = paginator.paginate(
            self.db.session.query(Message).filter(
                Message.conversation_id == conversation.id,
                Message.status.in_([MessageStatus.STOP, MessageStatus.NORMAL]),
                Message.answer != "",
                ~Message.is_deleted,
                *filters,
            ).order_by(desc("created_at"))
        )

        return messages, paginator

    def delete_conversation(self, conversation_id: UUID, account: Account) -> Conversation:
        """根据传递的会话id+账号删除指定的会话记录"""
        # 1.获取会话记录并校验权限
        conversation = self.get_conversation(conversation_id, account)

        # 2.更新会话的删除状态
        self.update(conversation, is_deleted=True)

        return conversation

    def delete_message(self, conversation_id: UUID, message_id: UUID, account: Account) -> Message:
        """根据传递的会话id+消息id删除指定的消息记录"""
        # 1.获取会话记录并校验权限
        conversation = self.get_conversation(conversation_id, account)

        # 2.获取消息并校验权限
        message = self.get_message(message_id, account)

        # 3.判断消息和会话是否关联
        if conversation.id != message.conversation_id:
            raise NotFoundException("该会话下不存在该消息，请核实后重试")

        # 4.校验通过修改消息is_deleted属性标记删除
        self.update(message, is_deleted=True)

        return message

    def update_conversation(self, conversation_id: UUID, account: Account, **kwargs) -> Conversation:
        """根据传递的会话id+账号+kwargs更新会话信息"""
        # 1.获取会话记录并校验权限
        conversation = self.get_conversation(conversation_id, account)

        # 2.更新会话信息
        self.update(conversation, **kwargs)

        return conversation
