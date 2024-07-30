import asyncio

import pygame
from loguru import logger

from paper_audio.audio.processor import AsyncTTSProcessor, AudioSegment


class AsyncAudioPlayer:
    def __init__(self, tts_processor: AsyncTTSProcessor):
        self.tts_processor = tts_processor
        
        self.buffer = asyncio.Queue(maxsize=5)
        self.audio_segments = {}
        self.next_segment_to_play = 0

        self.playing = asyncio.Event()
        self.stopped = asyncio.Event()
        self.paused = asyncio.Event()

        self.current_sound = None
        self.current_id = None
        self.current_paragraph = None
        self.current_paragraph_simplified = None
        self.current_sentence = None

        self.init_pygame()

    def init_pygame(self):
        try:
            pygame.init()
            pygame.mixer.init()
        except pygame.error:
            logger.error("Error initializing Pygame. Audio playback may not work.")

    @property
    def is_ready(self) -> bool:
        return self.playing.is_set()
    
    @property
    def is_paused(self) -> bool:
        return self.paused.is_set()

    @property
    def is_stopped(self) -> bool:
        return self.stopped.is_set()

    async def collect_audio(self):
        while not self.is_stopped:
            try:
                segment: AudioSegment = await asyncio.wait_for(
                    self.tts_processor.buffer.get(), timeout=1.0
                )
                self.audio_segments[segment.rank] = segment
                await self.buffer.put(segment)
                self.tts_processor.buffer.task_done()
                logger.info(f"Received audio: {segment.rank=}")
            except asyncio.TimeoutError:
                continue

    async def play_audio(self):
        self.playing.set()
        while not self.is_stopped:
            if self.is_paused or (
                self.next_segment_to_play not in self.audio_segments
            ):
                await asyncio.sleep(0.1)
                continue

            await self.playing.wait()
            
            segment = self.audio_segments[self.next_segment_to_play]
            
            try:
                logger.info(f"Playing audio for {self.next_segment_to_play}: {segment.audio_path}")
                
                self.current_sound = pygame.mixer.Sound(segment.audio_path)
                self.current_sound.play()
                
                try:
                    await asyncio.wait_for(
                        self.buffer.get(), timeout=1.0
                    )
                    self.buffer.task_done()
                except asyncio.TimeoutError:
                    pass

                self.current_id = self.next_segment_to_play
                self.current_paragraph = segment.original_paragraph
                self.current_paragraph_simplified = segment.simplified_paragraph
                self.current_sentence = segment.sentence

                while pygame.mixer.get_busy() and not self.is_stopped:
                    await asyncio.sleep(0.1)
            
            except pygame.error as e:
                logger.error(f"Error playing audio: {e}")

            logger.info(f"Finishing audio for {self.current_id}")
            self.next_segment_to_play += 1
        
        self.current_sound = None
        self.current_id = None
        self.current_paragraph = None
        self.current_paragraph_simplified = None
        self.current_sentence = None
        self.playing.clear()

    async def run(self):
        try:
            generate_task = asyncio.create_task(self.tts_processor.process_paragraphs())
            collect_task = asyncio.create_task(self.collect_audio())
            play_task = asyncio.create_task(self.play_audio())
            await asyncio.gather(generate_task, collect_task, play_task)
        finally:
            await self.stop()

    async def pause(self):
        self.paused.set()
        if self.current_sound and pygame.mixer.get_busy():
            pygame.mixer.pause()
            logger.info(f"Paused at {self.current_id=}")

    async def resume(self):
        self.paused.clear()
        if self.current_sound:
            logger.info(f"Resumed from {self.current_id=}")
            pygame.mixer.unpause()

    async def stop(self):
        self.stopped.set()
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        pygame.quit()

    def set_paragraphs(self, paragraphs: list[str]):
        self.next_segment_to_play = 0
        self.audio_segments.clear()
        self.tts_processor.paragraphs = paragraphs
