import hashlib
import json
import os
from asyncio.queues import Queue
from dataclasses import dataclass

import aiohttp
import redis.asyncio as redis
import spacy
from loguru import logger

nlp = spacy.load("en_core_web_sm")

@dataclass
class AudioSegment:
    rank: int
    paragraph_id: int
    sentence_id: int
    sentence: str
    original_paragraph: str
    simplified_paragraph: str
    audio_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "rank": self.rank,
                "paragraph_id": self.paragraph_id,
                "sentence_id": self.sentence_id,
                "sentence": self.sentence,
                "original_paragraph": self.original_paragraph,
                "simplified_paragraph": self.simplified_paragraph,
                "audio_path": self.audio_path,
            }
        )

    @property
    def key(self) -> str:
        return hashlib.md5(self.to_json().encode()).hexdigest()

    @classmethod
    def from_json(cls, json_str: str) -> "AudioSegment":
        data = json.loads(json_str)
        return cls(
            rank=data["rank"],
            paragraph_id=data["paragraph_id"],
            sentence_id=data["sentence_id"],
            sentence=data["sentence"],
            original_paragraph=data["original_paragraph"],
            simplified_paragraph=data["simplified_paragraph"],
            audio_path=data.get("audio_path", None),
        )


class AsyncTTSProcessor:
    def __init__(
        self,
        tts_api_key: str,
        chat_api_key: str,
        tts_end_point: str = "https://api.openai.com/v1/audio/speech",
        chat_end_point: str = "https://api.groq.com/openai/v1/chat/completions",
        tts_model: str = "tts-1",
        chat_model: str = "llama-3.1-8b-instant",
        voice: str = "alloy",
        speed: int = 1.0,
        redis_url: str = "redis://localhost",
        buffer_size: int = 2,
        cache_dir: str = "audio_cache",
        cache_ttl: int = 60 * 60 * 24 * 30,
    ):
        self.buffer = Queue(maxsize=buffer_size)
        self.audio_cache_dir = cache_dir
        self.tts_api_key = tts_api_key
        self.chat_api_key = chat_api_key
        self.session = None
        self.redis = None
        self.redis_url = redis_url
        self.voice = voice
        self.speed = speed

        self.tts_end_point = tts_end_point
        self.chat_end_point = chat_end_point
        self.tts_model = tts_model
        self.chat_model = chat_model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.cache_ttl = cache_ttl

        self.paragraphs = []

        if not os.path.exists(self.audio_cache_dir):
            os.makedirs(self.audio_cache_dir)

    async def initialize(self):
        self.redis = redis.from_url(
            self.redis_url, encoding="utf-8", decode_responses=True
        )

    async def close(self):
        if self.redis:
            await self.redis.aclose()

    async def tts(self, paragraph: str) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.tts_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.tts_model,
            "input": paragraph,
            "voice": self.voice,
        }

        async with self.session.post(
            self.tts_end_point, headers=headers, json=data
        ) as response:
            if response.status == 200:
                audio_content = await response.read()
                return audio_content
            else:
                raise Exception(
                    f"API request failed with status {response.status}: {await response.text()}"
                )

    async def simplify(self, paragraph: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.chat_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": "You are a helpful accessibility assistant. Your job is to make the user input (academic text) more suitable to be used with text to speech.\n\nSkip any trivial information. Do not add any information that is not present. Do not return anything else. Just the resulting text."},
                {"role": "user", "content": "InstructGPT"},
                {"role": "assistant", "content": "Instruct G P T"},
                {
                    "role": "user",
                    "content": "Scalable Extraction of Training Data from (Production) Language Models",
                },
                {
                    "role": "assistant",
                    "content": "Scalable Extraction of Training Data from Language Models",
                },
                {
                    "role": "user",
                    "content": "B =15\nx+B\ni=x\n ℓi−\n1\nx+B\nj=x\nℓj  2\n.\n8",
                },
                {"role": "assistant", "content": "Skipped"},
                {
                    "role": "user",
                    "content": "Here ℓis the signal of logits and xthe index. Using this new signal we compute variances again but this time from the\npoint xto the end of the sequence\nVarEndB [ℓ](x)= 1\nS−x\nS\ni=x\n VarWinB [ℓ](i)−\nS−x\nS\nj=x\nVarWinB [ℓ](i) 2\n.\nIf this signal drops below a certain threshold (we choose 6.75) and stays below for the remainder of the sequence, we\nclassify the sequence to have repetitions.",
                },
                {
                    "role": "assistant",
                    "content": "Here l is the signal of logits and x the index. Using this new signal we compute variances again but this time from the point x to the end of the sequence.\nIf this signal drops below a certain threshold (we choose 6.75) and stays below for the remainder of the sequence, we classify the sequence to have repetitions.",
                },
                {"role": "user", "content": f"{paragraph}"},
            ],
            "max_tokens": 300,
            "n": 1,
            "stop": None,
            "temperature": 0.2,
        }

        try:
            async with self.session.post(
                self.chat_end_point, headers=headers, json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result["choices"][0]["message"]["content"].strip()
                    prompt_tokens = result["usage"]["prompt_tokens"]
                    completion_tokens = result["usage"]["completion_tokens"]
                    return summary, prompt_tokens, completion_tokens
                else:
                    raise Exception(
                        f"API request failed with status {response.status}: {await response.text()}"
                    )

        except Exception as e:
            return f"An error occurred: {str(e)}", 0, 0

    async def add_to_cache(self, segment: AudioSegment, audio_content: bytes) -> str:
        audio_path = os.path.join(self.audio_cache_dir, f"{segment.key}.mp3")
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_content)
        segment.audio_path = audio_path
        await self.redis.set(
            segment.key,
            segment.to_json(),
        )
        await self.redis.expire(segment.key, self.cache_ttl)

        return audio_path

    async def get_from_cache(self, key: str) -> AudioSegment | None:
        cached_data = await self.redis.get(key)
        if not cached_data:
            return None
        return AudioSegment.from_json(cached_data)

    async def process_sentence(self, segment: AudioSegment):
        try:
            cached_path = await self.get_from_cache(segment.key)
            if not cached_path:
                audio_content = await self.tts(segment.sentence)
                cached_path = await self.add_to_cache(
                    segment, audio_content=audio_content
                )
            segment.audio_path = cached_path

        except Exception as e:
            logger.error(
                f"Error processing segment: {segment.sentence_id}/{segment.paragraph_id} - {str(e)}"
            )
            return

        await self.buffer.put(segment)

    async def process_paragraph(self, rank: int, idx: int, paragraph: str):
        paragraph_key = hashlib.md5(paragraph.encode()).hexdigest()
        cached_summary = await self.redis.get(f"p_{paragraph_key}")
        if cached_summary is None:
            cached_summary, prompt_tokens, completion_tokens = await self.simplify(
                paragraph
            )
            await self.redis.set(f"p_{paragraph_key}", cached_summary)
            await self.redis.expire(f"p_{paragraph_key}", self.cache_ttl)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

        for j, sentence in enumerate(nlp(cached_summary).sents):
            sentence = sentence.text.strip()
            await self.process_sentence(
                AudioSegment(
                    rank=rank,
                    paragraph_id=idx,
                    sentence_id=j,
                    sentence=sentence,
                    original_paragraph=paragraph,
                    simplified_paragraph=cached_summary,
                )
            )
            rank += 1

        return rank

    async def process_paragraphs(self, paragraphs: list[str] | None = None):
        if paragraphs is None:
            paragraphs = self.paragraphs

        await self.initialize()
        rank = 0
        async with aiohttp.ClientSession() as self.session:
            for i, paragraph in enumerate(paragraphs):
                logger.info(f"Processing [{i+1}/{len(paragraphs)}] {rank=}")
                rank = await self.process_paragraph(rank, i, paragraph)

        await self.close()
