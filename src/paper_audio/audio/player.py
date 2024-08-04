import asyncio

import pygame
from loguru import logger

from paper_audio.audio.processor import AsyncTTSProcessor, AudioSegment


class AsyncAudioPlayer:
    def __init__(self, producer: AsyncTTSProcessor):
        self.producer = producer
        
        self.buffer = asyncio.Queue(maxsize=2)
        self.audio_segments = {}
        self.next_segment_to_play = 0

        self.playing = asyncio.Event()
        self.stopped = asyncio.Event()
        self.paused = asyncio.Event()

        self.curr_playback = None
        self.curr_rank = None
        self.curr_orig_paragraph = None
        self.curr_simp_paragraph = None
        self.curr_sentence = None

        self.total_chars = 0

        self.init_pygame()

    def init_pygame(self):
        try:
            pygame.init()
            pygame.mixer.init()
        except pygame.error:
            logger.error("Error initializing Pygame. Audio playback may not work.")
    
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
                    self.producer.buffer.get(), timeout=1.0
                )
                self.audio_segments[segment.rank] = segment
                # This adds limit to the player
                await self.buffer.put(segment)
                self.producer.buffer.task_done()
                self.total_chars += len(segment.sentence)
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
                logger.info(f"Playing audio for {self.next_segment_to_play}")
                self.curr_playback = pygame.mixer.Sound(segment.audio_path)
                self.curr_playback.play()
                
                try:
                    await asyncio.wait_for(
                        self.buffer.get(), timeout=1.0
                    )
                    self.buffer.task_done()
                except asyncio.TimeoutError:
                    pass

                self.curr_rank = self.next_segment_to_play
                self.curr_orig_paragraph = segment.original_paragraph
                self.curr_simp_paragraph = segment.simplified_paragraph
                self.curr_sentence = segment.sentence

                while pygame.mixer.get_busy() and not self.is_stopped:
                    await asyncio.sleep(0.1)
            
            except pygame.error as e:
                logger.error(f"Error playing audio: {e}")

            logger.info(f"Finishing audio for {self.curr_rank}")
            self.next_segment_to_play += 1

            if self.curr_rank in self.audio_segments:
                del self.audio_segments[self.curr_rank]
        
        self.curr_playback = None
        self.curr_rank = None
        self.curr_orig_paragraph = None
        self.curr_simp_paragraph = None
        self.curr_sentence = None
        self.playing.clear()

    async def run(self):
        try:
            generate_task = asyncio.create_task(self.producer.process_paragraphs())
            collect_task = asyncio.create_task(self.collect_audio())
            play_task = asyncio.create_task(self.play_audio())
            await asyncio.gather(generate_task, collect_task, play_task)
        finally:
            await self.stop()

    async def pause(self):
        self.paused.set()
        if self.curr_playback and pygame.mixer.get_busy():
            pygame.mixer.pause()
            logger.info(f"Paused at {self.curr_rank=}")

    async def resume(self):
        self.paused.clear()
        if self.curr_playback:
            pygame.mixer.unpause()
            logger.info(f"Resumed from {self.curr_rank=}")

    async def stop(self):
        self.stopped.set()
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        pygame.quit()

        self.audio_segments.clear()
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
                self.buffer.task_done()
            except asyncio.QueueEmpty:
                break

    def set_paragraphs(self, paragraphs: list[str]):
        self.next_segment_to_play = 0
        self.audio_segments.clear()
        self.producer.paragraphs = paragraphs
