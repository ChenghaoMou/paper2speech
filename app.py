import asyncio
import os

import gradio as gr
from dotenv import load_dotenv
from paper_audio.audio.player import AsyncAudioPlayer
from paper_audio.audio.processor import AsyncTTSProcessor
from paper_audio.text import get_paragraphs, get_text

load_dotenv()

class GradioInterface:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.tts_processor = AsyncTTSProcessor(api_key=self.api_key)
        self.player = AsyncAudioPlayer(self.tts_processor)
        self.player_task = None
    
    def reset(self):
        self.tts_processor = AsyncTTSProcessor(api_key=self.api_key)
        self.player = AsyncAudioPlayer(self.tts_processor)

    async def start(self, file_path: str):
        if self.player_task:
            await self.stop()
        
        self.reset()
        paragraphs = list(get_paragraphs(get_text(file_path)))
        self.player.set_paragraphs(paragraphs)
        self.player_task = asyncio.create_task(self.player.run())
        return "Started processing paragraphs"

    async def pause(self):
        await self.player.pause()
        return """**Paused**"""

    async def resume(self):
        await self.player.resume()
        return """**Resumed**"""

    async def stop(self):
        if self.player_task:
            await self.player.stop()
            await self.player_task
            self.player_task = None
            self.reset()
        return """**Stopped**"""

    def display_function(self):
        
        while True:
            if self.player.current_id is None:
                continue

            curr_sentence = self.player.current_sentence.strip(' ')
            curr_para = self.player.current_paragraph_simplified
            text_with_highlight = curr_para.replace(curr_sentence, f" **{curr_sentence}** ", 1)

            yield f"""
    ## Simplified Paragraph

    {text_with_highlight}

    ## Original Paragraph

    > {self.player.current_paragraph}
    """

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Monochrome(
            font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"],
            text_size=gr.themes.sizes.text_lg,
        )) as interface:
            gr.Markdown("# Paper to Speech Player")

            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload PDF")
                    start_btn = gr.Button("Start")
                    pause_btn = gr.Button("Pause")
                    resume_btn = gr.Button("Resume")
                    stop_btn = gr.Button("Stop")

                    status = gr.Markdown(label="Status")

                with gr.Column(scale=4):
                    paragraph_display = gr.Markdown(label="Details")
                    interface.load(self.display_function, outputs=[paragraph_display], show_progress=False)

            start_btn.click(self.start, inputs=[file_input], outputs=[status])
            pause_btn.click(self.pause, outputs=[status])
            resume_btn.click(self.resume, outputs=[status])
            stop_btn.click(self.stop, outputs=[status])

        return interface

async def main():
    gradio_interface = GradioInterface()
    interface = gradio_interface.create_interface()
    await interface.launch()

if __name__ == "__main__":
    asyncio.run(main())