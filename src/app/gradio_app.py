"""Phase 8: Gradio web interface for the Nepal RAG Agent."""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from src.utils import get_settings

logger = logging.getLogger(__name__)


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    settings = get_settings()
    
    # Custom CSS for Nepali-friendly styling
    custom_css = """
    .gradio-container { font-family: 'Noto Sans Devanagari', 'Mangal', sans-serif; }
    .input-box textarea { font-size: 16px !important; }
    .output-box { font-size: 16px !important; line-height: 1.6; }
    """
    
    with gr.Blocks(
        title=settings["app"]["title"],
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"# {settings['app']['title']}")
        gr.Markdown(settings["app"]["description"])
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input
                query_input = gr.Textbox(
                    label="प्रश्न / Question",
                    placeholder="यहाँ आफ्नो प्रश्न लेख्नुहोस्... / Type your question here...",
                    lines=3,
                    elem_classes=["input-box"],
                )
                
                with gr.Row():
                    submit_btn = gr.Button("प्रश्न सोध्नुहोस् / Ask", variant="primary")
                    clear_btn = gr.Button("खाली गर्नुहोस् / Clear")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["नागरिकता कार्ड कसरी लिन सकिन्छ?"],
                        ["जग्गा नामसारी कसरी गर्ने?"],
                        ["वैदेशिक रोजगार अनुमोदन कसरी लिने?"],
                        ["स्वास्थ्य बीमा कसरी लिने?"],
                        ["बृद्धा भत्ता कसरी पाइन्छ?"],
                        ["SSF मा दर्ता कसरी गर्ने?"],
                    ],
                    inputs=query_input,
                )
            
            with gr.Column(scale=3):
                # Output
                answer_output = gr.Textbox(
                    label="उत्तर / Answer",
                    lines=10,
                    elem_classes=["output-box"],
                    interactive=False,
                )
                
                domain_output = gr.Textbox(
                    label="पहिचान गरिएको डोमेन / Identified Domain",
                    interactive=False,
                )
                
                sources_output = gr.JSON(
                    label="स्रोतहरू / Sources",
                )
        
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=[answer_output, domain_output, sources_output],
        )
        
        query_input.submit(
            fn=process_query,
            inputs=query_input,
            outputs=[answer_output, domain_output, sources_output],
        )
        
        clear_btn.click(
            fn=lambda: ("", "", {}, ""),
            outputs=[query_input, answer_output, sources_output, domain_output],
        )
    
    return demo


def process_query(query: str) -> tuple[str, str, dict[str, Any], str]:
    """Process a query and return results."""
    if not query or not query.strip():
        return ("कृपया प्रश्न लेख्नुहोस् / Please enter a question", "", {}, "")
    
    try:
        from src.generation.chat import ChatSession
        settings = get_settings()
        
        chat = ChatSession(
            provider=settings["generation"]["provider"],
            model=settings["generation"]["model"],
        )
        
        response = chat.ask(query)
        
        # Format sources
        sources = {}
        for cite in response.citations:
            sources[f"[{cite['index']}] {cite['name']}"] = cite["url"]
        
        domain_display = response.domain_id.replace("_", " ").title() if response.domain_id else "Unknown"
        
        return (
            response.answer,
            sources,
            domain_display,
        )
    except Exception as e:
        logger.error("Error processing query: %s", e)
        return (
            f"Error: {str(e)}",
            "",
            {},
        )


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio app."""
    demo = create_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_app()
