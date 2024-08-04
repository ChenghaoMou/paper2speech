import asyncio
import os

import gradio as gr
from dotenv import load_dotenv
from paper_audio.audio.player import AsyncAudioPlayer
from paper_audio.audio.processor import AsyncTTSProcessor
from paper_audio.text import get_text

load_dotenv()

class GradioInterface:
    def __init__(self):
        self.tts_api_key = os.environ["OPENAI_API_KEY"]
        self.chat_api_key = os.environ["GROQ_API_KEY"]
        self.producer = AsyncTTSProcessor(tts_api_key=self.tts_api_key, chat_api_key=self.chat_api_key)
        self.player = AsyncAudioPlayer(self.producer)
        self.player_task = None
    
    def reset(self):
        self.producer = AsyncTTSProcessor(tts_api_key=self.tts_api_key, chat_api_key=self.chat_api_key)
        self.player = AsyncAudioPlayer(self.producer)

    async def start(self, file_path: str):
        if self.player_task:
            await self.stop()
        
        self.reset()
        paragraphs = get_text(file_path)
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

    def render_simplified(self):
        
        while True:
            if self.player.curr_rank is None:
                continue

            curr_sentence = self.player.curr_sentence.strip(' ')
            curr_para = self.player.curr_simp_paragraph
            text_with_highlight = curr_para.replace(curr_sentence, f" <u>{curr_sentence}</u> ", 1)

            yield f"""{text_with_highlight}"""

    def render_original(self):
        while True:
            if self.player.curr_rank is None:
                continue

            yield f"""

    ## Original Paragraph

    > {self.player.curr_orig_paragraph}
    """

    def render_details(self):
        while True:
            tts_count = self.player.total_chars
            tts_cost = tts_count / 1_000_000 * 15

            chat_cost = self.producer.total_prompt_tokens / 1_000_000 * 0.15 + self.producer.total_completion_tokens / 1_000_000 * 0.075

            yield f"""
- Total characters: {self.player.total_chars}
- Estimated cost: ${tts_cost:.4f}
- Estimated completion cost: ${chat_cost:.4f}
- Estimated total cost: ${tts_cost + chat_cost:.4f}
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
                    pause_btn = gr.Button("⏸️️")
                    resume_btn = gr.Button("▶️️")
                    stop_btn = gr.Button("⏹️️")
                    status = gr.Markdown(label="Status")
                    details_display = gr.Markdown(label="Cost")
                    interface.load(self.render_details, outputs=[details_display], show_progress=False)

                with gr.Column(scale=2):
                    with gr.Tab("Simplified"):
                        with gr.Row():
                            paragraph_display = gr.Markdown()
                            interface.load(self.render_simplified, outputs=[paragraph_display], show_progress=False)
                    with gr.Tab("Original"):
                        details_display = gr.Markdown()
                        interface.load(self.render_original, outputs=[details_display], show_progress=False)

            file_input.change(self.start, inputs=[file_input], outputs=[status])
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