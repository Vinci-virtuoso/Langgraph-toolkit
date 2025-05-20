"""
LangGraphAdapter masquerades as a livekit.LLM and translates the LiveKit chat chunks
into LangGraph messages.
"""

from typing import Any, Optional, Dict
from livekit.agents import llm
from langgraph.pregel import PregelProtocol
from langchain_core.messages import BaseMessageChunk, AIMessage, HumanMessage
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import SynthesizeStream
from livekit.agents.utils import shortuuid
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from httpx import HTTPStatusError
import logging

logger = logging.getLogger(__name__)


# https://github.com/livekit/agents/issues/1370#issuecomment-2588821571
class FlushSentinel(str, SynthesizeStream._FlushSentinel):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class LangGraphStream(llm.LLMStream):
    def __init__(
        self,
        llm: llm.LLM,
        chat_ctx: llm.ChatContext,
        graph: PregelProtocol,
        tools: list[llm.FunctionTool] = None,
        conn_options: APIConnectOptions = None,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._graph = graph

    async def _run(self):
        # Change: take the last human message from both chat_ctx.messages and chat_ctx.items
        input_human_message = next(
            (
                self._to_message(m)
                for m in reversed(self.chat_ctx.items)
                if m.role == "user"
            ),
            None,
        )

        messages = [input_human_message] if input_human_message else []
        input_data = {"messages": messages}

        # Check for interrupts
        if interrupt := await self._get_interrupt():
            used_messages = [AIMessage(interrupt.value), input_human_message]
            input_data = Command(resume=(input_human_message.content, used_messages))
        sent_response = False  # Flag to ensure we only send one response
        try:
            async for mode, data in self._graph.astream(
                input_data, config=self._llm._config, stream_mode=["messages", "custom"]
            ):
                if mode == "messages" and not sent_response:
                    if chunk := await self._to_livekit_chunk(data[0]):
                        self._event_ch.send_nowait(chunk)
                        sent_response = True
                elif mode == "custom":
                    if isinstance(data, dict) and (event := data.get("type")):
                        if event == "say" and not sent_response:
                            content = (data.get("data") or {}).get("content")
                            if chunk := await self._to_livekit_chunk(content):
                                self._event_ch.send_nowait(chunk)
                                sent_response = True
                        elif event == "flush":
                            self._event_ch.send_nowait(
                                self._create_livekit_chunk(FlushSentinel())
                            )
        except GraphInterrupt:
            pass

        # If interrupted, send the interrupt value as a message
        if interrupt := await self._get_interrupt():
            if chunk := await self._to_livekit_chunk(interrupt.value):
                self._event_ch.send_nowait(chunk)

    async def _get_interrupt(self) -> Optional[str]:
        try:
            state = await self._graph.aget_state(config=self._llm._config)
            interrupts = [interrupt for task in state.tasks for interrupt in task.interrupts]
            assistant = next(
                (interrupt for interrupt in reversed(interrupts) if isinstance(interrupt.value, str)), None
            )
            return assistant
        except (HTTPStatusError, ValueError) as e:
            # If using a local graph without a checkpointer, just ignore interrupts.
            if isinstance(e, ValueError) and str(e) == "No checkpointer set":
                return None
            raise e

    def _to_message(self, msg: llm.ChatMessage) -> HumanMessage:
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = []
            for c in msg.content:
                if isinstance(c, str):
                    content.append({"type": "text", "text": c})
                elif hasattr(c, "image") and isinstance(c.image, str):
                    content.append({"type": "image_url", "image_url": c.image})
                else:
                    logger.warning("Unsupported content type")
        else:
            content = ""
        return HumanMessage(content=content, id=msg.id)

    @staticmethod
    def _create_livekit_chunk(content: str, *, id: Optional[str] = None) -> llm.ChatChunk | None:
        return llm.ChatChunk(
            id=id or shortuuid(),
            delta=llm.ChoiceDelta(role="assistant", content=content),
        )

    @staticmethod
    async def _to_livekit_chunk(msg: BaseMessageChunk | str | None) -> llm.ChatChunk | None:
        if not msg:
            return None

        request_id = None
        content = msg

        if isinstance(msg, str):
            content = msg
        elif hasattr(msg, "content") and isinstance(msg.content, str):
            request_id = getattr(msg, "id", None)
            content = msg.content
        elif isinstance(msg, dict):
            request_id = msg.get("id")
            content = msg.get("content")

        return LangGraphStream._create_livekit_chunk(content, id=request_id)


class LangGraphAdapter(llm.LLM):
    def __init__(self, graph: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ) -> llm.LLMStream:
        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            graph=self._graph,
            tools=tools,
            conn_options=conn_options,
        )