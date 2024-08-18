import gradio as gr
from langflow import LangflowPipeline
from langsmith import Client
from jina import Document, DocumentArray
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith client
langsmith_client = Client()

# Initialize Langflow pipeline
pipeline = LangflowPipeline.from_json("langflow_pipeline.json")

async def scrape_url(url):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                return text
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""

async def scrape_and_process(url):
    try:
        # Scrape the main URL
        main_content = await scrape_url(url)

        # Extract sub-links
        soup = BeautifulSoup(main_content, 'html.parser')
        sub_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(url)]

        # Scrape sub-links
        tasks = [scrape_url(sub_url) for sub_url in sub_links[:5]]  # Limit to 5 sub-links for demo purposes
        sub_contents = await asyncio.gather(*tasks)

        # Combine all content
        all_content = main_content + ' '.join(sub_contents)

        # Process with Jina
        doc = Document(text=all_content)
        docs = DocumentArray([doc])
        processed_docs = docs.apply(pipeline)

        # Use LangSmith to track and analyze the results
        with langsmith_client.track_run("knowledge_scraping") as run:
            run.inputs = {"url": url, "content_length": len(all_content)}
            run.outputs = {"processed_docs": processed_docs.to_dict()}

        return processed_docs[0].text
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return f"An error occurred: {str(e)}"

def scrape_and_display(url):
    result = asyncio.run(scrape_and_process(url))
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=scrape_and_display,
    inputs=gr.Textbox(label="Enter URL"),
    outputs=gr.Markdown(label="Extracted Knowledge"),
    title="URL Knowledge Scraper",
    description="Enter a URL to scrape and distill knowledge from the page and its sub-pages.",
    theme="huggingface",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
