import gradio as gr
from langflow import LangflowPipeline
from langsmith import Client
import aiohttp
import asyncio
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
from datetime import datetime
from keybert import KeyBERT
import sqlite3
import urllib.parse
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from treelib import Tree
import plotly.graph_objects as go
import re
import os
import tempfile
import subprocess
from github import Github
from github.GithubException import GithubException
import requests
from packaging import version
import pkg_resources
import dedupe
from sqlite_utils import Database
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients and models
langsmith_client = Client()
pipeline = LangflowPipeline.from_json("langflow_pipeline.json")
key_bert_model = KeyBERT()
openai_client = AsyncOpenAI()

# Initialize GitHub client
github_client = Github(os.environ.get('GITHUB_ACCESS_TOKEN'))

# Database setup
conn = sqlite3.connect('knowledge_base.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS knowledge_tree
             (id TEXT PRIMARY KEY, parent_id TEXT, title TEXT, content TEXT, last_updated TIMESTAMP)''')
conn.commit()

# Initialize dedupe
deduper = dedupe.Dedupe([
    {'field': 'title', 'type': 'String'},
    {'field': 'content', 'type': 'Text'},
])

# Initialize TfidfVectorizer for content similarity
vectorizer = TfidfVectorizer(stop_words='english')

def is_github_url(url):
    github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
    return re.match(github_pattern, url) is not None

async def fetch_jina_reader_content(url):
    jina_url = f"https://r.jina.com/{urllib.parse.quote(url)}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(jina_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching from Jina Reader API: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching from Jina Reader API: {str(e)}")
            return None

async def fetch_pypi_docs(package_name):
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        response.raise_for_status()
        package_info = response.json()

        latest_version = max(package_info['releases'].keys(), key=version.parse)
        
        doc_url = package_info['info'].get('project_urls', {}).get('Documentation')
        if not doc_url:
            doc_url = package_info['info'].get('home_page')

        readme_content = package_info['info'].get('description', '')

        return {
            'name': package_name,
            'version': latest_version,
            'doc_url': doc_url,
            'readme': readme_content
        }
    except Exception as e:
        logger.error(f"Error fetching PyPI docs for {package_name}: {str(e)}")
        return None

def parse_requirements(repo_path):
    requirements_file = os.path.join(repo_path, 'requirements.txt')
    if not os.path.exists(requirements_file):
        return []

    try:
        with open(requirements_file, 'r') as f:
            return [
                pkg_resources.Requirement.parse(line.strip()).name
                for line in f
                if line.strip() and not line.startswith('#')
            ]
    except Exception as e:
        logger.error(f"Error parsing requirements.txt: {str(e)}")
        return []

async def process_github_repo(url, fetch_docs=False):
    _, _, _, owner, repo = url.split('/')

    try:
        repo = github_client.get_repo(f"{owner}/{repo}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            clone_url = repo.clone_url
            subprocess.run(['git', 'clone', clone_url, tmp_dir], check=True)

            repo_tree = create_repo_tree(tmp_dir, repo.name)

            if fetch_docs:
                requirements = parse_requirements(tmp_dir)
                docs_tree = Tree()
                docs_root_id = f"docs_{repo.name}"
                docs_tree.create_node(f"Documentation for {repo.name}", docs_root_id)

                for package in requirements:
                    docs = await fetch_pypi_docs(package)
                    if docs:
                        package_node_id = f"docs_{repo.name}_{package}"
                        docs_tree.create_node(package, package_node_id, parent=docs_root_id, data=docs)

                repo_tree.paste(repo_tree.root, docs_tree)

            return repo_tree

    except GithubException as e:
        logger.error(f"GitHub API error: {str(e)}")
        return None

def create_repo_tree(repo_path, repo_name):
    tree = Tree()
    root_id = f"github_{repo_name}"
    tree.create_node(repo_name, root_id)

    for root, dirs, files in os.walk(repo_path):
        if '.git' in dirs:
            dirs.remove('.git')

        relative_path = os.path.relpath(root, repo_path)
        if relative_path == '.':
            parent = root_id
        else:
            parent = f"github_{repo_name}_{relative_path}"

        for dir in dirs:
            dir_path = os.path.join(relative_path, dir)
            tree.create_node(dir, f"github_{repo_name}_{dir_path}", parent=parent)

        for file in files:
            file_path = os.path.join(relative_path, file)
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            file_id = f"github_{repo_name}_{file_path}"
            tree.create_node(file, file_id, parent=parent, data={"content": content})

    return tree

async def process_url(url, max_depth=2, fetch_docs=False):
    try:
        if is_github_url(url):
            tree = await process_github_repo(url, fetch_docs)
            if tree is None:
                return {"error": "Failed to process GitHub repository"}

            store_tree_in_db(tree)

            result = {
                "title": f"GitHub Repository: {tree[tree.root].tag}",
                "full_text": "This is a GitHub repository. Please refer to the tree structure for details.",
                "summary": f"GitHub repository structure for {tree[tree.root].tag}",
                "keywords": [("github", 1.0), ("repository", 0.9), (tree[tree.root].tag, 0.8)],
                "graph": None,
                "tree": tree_to_plotly(tree)
            }
        else:
            jina_content = await fetch_jina_reader_content(url)

            if not jina_content:
                return {"error": "Failed to fetch content from Jina Reader API"}

            full_text = jina_content.get('text', '')
            summary = jina_content.get('summary', '')
            title = jina_content.get('title', '')
            
            keywords = key_bert_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

            graph = create_knowledge_graph(full_text, keywords)

            tree = Tree()
            root_id = "web_content"
            tree.create_node(title, root_id, data={"content": full_text, "summary": summary})

            for keyword, _ in keywords:
                tree.create_node(keyword, f"{root_id}_{keyword}", parent=root_id)

            store_tree_in_db(tree)

            result = {
                "title": title,
                "full_text": full_text,
                "summary": summary,
                "keywords": keywords,
                "graph": graph_to_image(graph),
                "tree": tree_to_plotly(tree)
            }

        return result
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return {"error": str(e)}

def store_tree_in_db(tree):
    def recursive_store(node, parent_id=None):
        node_id = node.identifier
        title = node.tag
        content = node.data.get('content', '') if node.data else ''

        # Check for duplicates
        if not is_duplicate(title, content):
            c.execute("INSERT OR REPLACE INTO knowledge_tree VALUES (?, ?, ?, ?, ?)", 
                      (node_id, parent_id, title, content, datetime.now()))

        for child in tree.children(node_id):
            recursive_store(child, node_id)

    recursive_store(tree[tree.root])
    conn.commit()

def is_duplicate(title, content):
    # Check for exact duplicates
    c.execute("SELECT id FROM knowledge_tree WHERE title = ? AND content = ?", (title, content))
    if c.fetchone():
        return True

    # Check for near-duplicates using dedupe
    data = [{'title': title, 'content': content}]
    deduper.match(data, threshold=0.9, canonicalize=True)

    return len(deduper.match(data, threshold=0.9, canonicalize=True)) > 0

def create_knowledge_graph(text, keywords):
    G = nx.Graph()
    for keyword, _ in keywords:
        G.add_node(keyword)
    for i, (keyword1, _) in enumerate(keywords):
        for keyword2, _ in keywords[i+1:]:
            if keyword1 in text and keyword2 in text:
                G.add_edge(keyword1, keyword2)
    return G

def graph_to_image(G):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold')
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'

def tree_to_plotly(tree):
    def trace_node(node, parent_id, x, y, width, height, trace):
        node_id = node.identifier
        trace['x'].append(x)
        trace['y'].append(y)
        trace['text'].append(node.tag)
        trace['parent'].append(parent_id)
        trace['node_ids'].append(node_id)

        children = tree.children(node_id)
        if children:
            child_width = width / len(children)
            for i, child in enumerate(children):
                child_x = x - width/2 + child_width/2 + i*child_width
                child_y = y - height
                trace_node(child, node_id, child_x, child_y, child_width, height, trace)

    trace = {
        'x': [],
        'y': [],
        'text': [],
        'parent': [],
        'node_ids': [],
        'type': 'scatter',
        'mode': 'markers+text',
        'marker': {'size': 10},
        'textposition': 'top center'
    }

    root = tree[tree.root]
    trace_node(root, '', 0, 0, 2, 0.5, trace)

    layout = {
        'hovermode': 'closest',
        'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
        'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False}
    }

    fig = go.Figure(data=[trace], layout=layout)
    return fig

def export_to_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_base64 = base64.b64encode(pdf_output.getvalue()).decode('utf-8')
    return f'data:application/pdf;base64,{pdf_base64}'

def scrape_and_display(url, fetch_docs):
    result = asyncio.run(process_url(url, fetch_docs=fetch_docs))
    tree = load_full_tree()
    tree_plot = tree_to_plotly(tree)
    return (
        result['title'],
        result['full_text'],
        result['summary'],
        ", ".join([k for k, _ in result['keywords']]),
        result.get('graph'),
        tree_plot,
        export_to_pdf(result['full_text'])
    )

def compare_urls(url1, url2):
    result1 = asyncio.run(process_url(url1))
    result2 = asyncio.run(process_url(url2))
    
    common_keywords = set([k for k, _ in result1['keywords']]) & set([k for k, _ in result2['keywords']])
    comparison = f"Common keywords: {', '.join(common_keywords)}\n\n"
    comparison += f"Summary of {url1}:\n{result1['summary']}\n\n"
    comparison += f"Summary of {url2}:\n{result2['summary']}"
    
    return comparison

def load_full_tree():
    tree = Tree()
    c.execute("SELECT id, parent_id, title, content FROM knowledge_tree")
    nodes = c.fetchall()
    for node in nodes:
        node_id, parent_id, title, content = node
        if parent_id is None:
            tree.create_node(title, node_id, data={"content": content})
        else:
            tree.create_node(title, node_id, parent=parent_id, data={"content": content})
    return tree
def export_tree_to_json(node_id="root"):
    def recursive_export(node_id):
        c.execute("SELECT title, content FROM knowledge_tree WHERE id = ?", (node_id,))
        node_data = c.fetchone()
        if not node_data:
            return None

        title, content = node_data
        node = {"title": title, "content": content, "children": []}

        c.execute("SELECT id FROM knowledge_tree WHERE parent_id = ?", (node_id,))
        children = c.fetchall()
        for child in children:
            child_node = recursive_export(child[0])
            if child_node:
                node["children"].append(child_node)

        return node

    return json.dumps(recursive_export(node_id), indent=2)

def consolidate_database():
    logger.info("Starting database consolidation...")
    
    # Get all entries
    c.execute("SELECT id, title, content FROM knowledge_tree")
    entries = c.fetchall()

    # Vectorize content for similarity comparison
    contents = [entry[2] for entry in entries]
    content_vectors = vectorizer.fit_transform(contents)

    # Calculate pairwise similarity
    similarity_matrix = cosine_similarity(content_vectors)

    # Find and merge similar entries
    merged_count = 0
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if similarity_matrix[i][j] > 0.9:  # Threshold for similarity
                merged_count += 1
                # Merge entries
                merged_content = f"{entries[i][2]}\n\n{entries[j][2]}"
                merged_title = f"{entries[i][1]} + {entries[j][1]}"
                
                # Update the first entry with merged content
                c.execute("UPDATE knowledge_tree SET title = ?, content = ? WHERE id = ?",
                          (merged_title, merged_content, entries[i][0]))
                
                # Delete the second entry
                c.execute("DELETE FROM knowledge_tree WHERE id = ?", (entries[j][0],))

    conn.commit()
    logger.info(f"Database consolidation complete. Merged {merged_count} entries.")

def prune_database():
    logger.info("Starting database pruning...")
    
    # Delete entries older than 6 months
    six_months_ago = datetime.now() - timedelta(days=180)
    c.execute("DELETE FROM knowledge_tree WHERE last_updated < ?", (six_months_ago,))
    
    deleted_count = c.rowcount
    conn.commit()
    
    logger.info(f"Database pruning complete. Deleted {deleted_count} old entries.")

def distill_knowledge():
    logger.info("Starting knowledge distillation...")
    
    # Get all entries
    c.execute("SELECT id, title, content FROM knowledge_tree")
    entries = c.fetchall()

    for entry in entries:
        entry_id, title, content = entry
        
        # Use OpenAI to generate a concise summary
        prompt = f"Summarize the following content in a concise manner:\n\n{content}\n\nSummary:"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes content."},
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Update the entry with the distilled content
        c.execute("UPDATE knowledge_tree SET content = ? WHERE id = ?", (summary, entry_id))

    conn.commit()
    logger.info("Knowledge distillation complete.")

# Schedule periodic tasks
scheduler = BackgroundScheduler()
scheduler.add_job(consolidate_database, IntervalTrigger(days=7))
scheduler.add_job(prune_database, IntervalTrigger(days=30))
scheduler.add_job(distill_knowledge, IntervalTrigger(days=14))
scheduler.start()

# Create Gradio interface
with gr.Blocks(theme="huggingface") as iface:
    gr.Markdown("# Advanced URL Knowledge Scraper with GitHub and PyPI Integration")
    with gr.Tab("Single URL"):
        url_input = gr.Textbox(label="Enter URL")
        fetch_docs_checkbox = gr.Checkbox(label="Fetch PyPI docs for GitHub repos", value=False)
        scrape_button = gr.Button("Scrape and Analyze")
        title_output = gr.Textbox(label="Title")
        with gr.Row():
            text_output = gr.Textbox(label="Extracted Text", lines=10)
            summary_output = gr.Textbox(label="Summary", lines=5)
        with gr.Row():
            keywords_output = gr.Textbox(label="Keywords")
            graph_output = gr.Image(label="Knowledge Graph")
        tree_output = gr.Plot(label="Tree of Knowledge")
        pdf_output = gr.File(label="Export as PDF")
        scrape_button.click(scrape_and_display, 
                            inputs=[url_input, fetch_docs_checkbox], 
                            outputs=[title_output, text_output, summary_output, keywords_output, graph_output, tree_output, pdf_output])
    
    with gr.Tab("Compare URLs"):
        url1_input = gr.Textbox(label="Enter first URL")
        url2_input = gr.Textbox(label="Enter second URL")
        compare_button = gr.Button("Compare")
        comparison_output = gr.Textbox(label="Comparison Result", lines=10)
        compare_button.click(compare_urls, inputs=[url1_input, url2_input], outputs=[comparison_output])

    with gr.Tab("Export Knowledge"):
        node_id_input = gr.Textbox(label="Enter Node ID (leave blank for entire tree)", placeholder="root")
        export_button = gr.Button("Export Knowledge")
        export_output = gr.File(label="Exported Knowledge File")
        export_button.click(export_tree_to_json, inputs=[node_id_input], outputs=[export_output])

    with gr.Tab("Database Management"):
        gr.Markdown("## Database Management")
        consolidate_button = gr.Button("Consolidate Database")
        prune_button = gr.Button("Prune Database")
        distill_button = gr.Button("Distill Knowledge")
        management_output = gr.Textbox(label="Management Result", lines=5)
        
        consolidate_button.click(consolidate_database, outputs=[management_output])
        prune_button.click(prune_database, outputs=[management_output])
        distill_button.click(distill_knowledge, outputs=[management_output])

if __name__ == "__main__":
    iface.launch()
