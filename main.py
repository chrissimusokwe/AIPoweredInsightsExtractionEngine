import os
import json
import time
import requests
import data_models as dm
import camelot
import pandas as pd
import numpy as np
import fitz # PyMuPDF
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tabulate import tabulate
from io import StringIO, BytesIO

# Load API keys from environment variables
load_dotenv()

#get the API keys from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# OpenAI and Groq Clients
# clientGRQ = Groq(api_key=GROQ_API_KEY)
clientOAI = OpenAI(api_key=OPENAI_API_KEY)

# Function to execute JS to access url
def execute_js_and_fetch_html(url):
    """Uses Selenium to render JavaScript and extract the final HTML."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    html = driver.page_source
    driver.quit()

    if not html.strip():
        raise ValueError("Fetched HTML content is empty")
    
    return html

# Function to parse HTML content and return text, tables, images, and hyperlinks
def parse_html(html_content, url):
    """Parses HTML content, extracting text, tables, images, and hyperlinks."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract and clean text
    for script in soup(["script", "style"]):
        script.extract()
    text = "\n".join([p.get_text() for p in soup.find_all("p")])
    
    # Extract tables
    tables = []
    for i, table in enumerate(pd.read_html(StringIO(html_content))):
        tables.append(dm.TableSchema(title=f"Table {i+1}", data=table.to_dict(orient="records")))
    
    # Remove table texts from paragraphs
    for table in soup.find_all("table"):
        table.extract()
    
    # Extract images
    images = []
    for img in soup.find_all("img", src=True):
        images.append(dm.ImageSchema(
            url=img["src"],
            description=img.get("alt", "No description available"),
            mine_site=url,  # Placeholder, might need LLM to extract mine site info
            category="Uncategorized"  # Placeholder, an LLM can classify
        ))
    
    # Extract hyperlinks
    hyperlinks = [a["href"] for a in soup.find_all("a", href=True)]
    
    return dm.ParsedDocument(text=text, tables=tables, hyperlinks=hyperlinks, images=images)

# Function to parse PDF content and return text, tables, images, and hyperlinks
def parse_pdf(pdf_content):
    """Parses PDF content, extracting text, tables, and images."""
    text_data = []
    text = ""
    tables = []
    formated_tables = []
    images = []

    data = BytesIO(pdf_content)
    
    """Extract plain text from PDF using PyMuPDF (fitz)."""
    doc = fitz.open(stream=data, filetype="pdf")
    for page in doc:
        text_data.append(page.get_text("text") or "")
    text = "\n".join(text_data)
    doc.close()
    
    """Extract tables using Camelot."""
    camelot_tables = camelot.read_pdf(data, pages="1-end", parallel = True, flavor="hybrid")
    for i, table in enumerate(camelot_tables):
        tables.append({"title": f"Camelot Table {i+1}", "data": table.df.to_dict(orient="records")})
        df = table.df  # Extract DataFrame
        camelot.plot(table, kind='grid').show()
        input("Press Enter to continue...")

    """Send extracted text and tables to an LLM for structured output."""
    prompt = f"""
    You are an expert in structuring unstructured data from PDFs.
    Your task: 
    - Structure the extracted data into a clean JSON format.
    - Ensure proper formatting of key-value pairs for structured output.
    - Correct any obvious errors in table formatting.
    Output the JSON result only.
    The following is the table data of a PDF file you are to structure:\n
    """

    #for table in tables:
        #response = clientGRQ.chat.completions.create(
        #    model="llama-3.1-8b-instant",
        #    messages=[{"role": "user", "content": prompt_complete}],
        #    temperature=0.3
        #)

        #formated_tables.append(json.loads(response.choices[0].message.content))

    # Extract images (Placeholder: needs a dedicated PDF image extraction approach)
    # Extract hyperlinks (Placeholder: needs a dedicated PDF hyperlink extraction approach)

    return dm.ParsedDocument(text=text, tables=tables, hyperlinks=[], images=images)

# Function to get URL and define if sends to parse HTML or PDF
def fetch_and_parse(url):
    response = requests.get(url)
    content_type = response.headers.get("Content-Type", "").lower()
    
    if "pdf" in content_type:
        return parse_pdf(response.content)
    else:
        html_content = execute_js_and_fetch_html(url)
        return parse_html(html_content, url)

# Function to chunk and embed text using Jina AI Embeddings API
def chunk_data(text: str) -> List[Dict]:
    start_time = time.time()
    
    # Define chunking parameters
    max_chunk_size = 1000  # characters
    min_chunk_size = 100   # characters
    
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_id = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph exceeds max size and we already have content,
        # save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk)
            })
            chunk_id += 1
            current_chunk = ""
        
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
            "char_count": len(current_chunk)
        })
    
    elapsed_time = time.time() - start_time
    print(f"[Chunk Data] Time: {elapsed_time:.2f}s, Chunks Created: {len(chunks)}")
    
    return chunks

# Function to label chunks with relevant topics
def label_chunks(chunks: List[str], topic_definitions: Dict[str, str]) -> List[Dict]:
    start_time = time.time()

    prompt = f"""Identify relevant topics in each chunk based on the following definitions:
    {json.dumps(topic_definitions, indent=2)}
    Return only the identified topic labels for each chunk in a structured JSON format."""
    
    response = clientOAI.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(chunks)}],
        response_format={"type": "json_schema",
                         "json_schema": {
                            "name":"chunks_schema",
                            "schema": dm.LabeledChunk.model_json_schema()
                         }
        }
    )

    labeled_chunks = json.loads(response.choices[0].message.content)
    #print("These are labeled chunks: " + json.dumps(labeled_chunks, indent=2))

    tokens_used = response.usage.total_tokens  # OpenAI API response includes token usage
    elapsed_time = time.time() - start_time
    print(f"[Label Chunks] Time: {elapsed_time:.2f}s, Tokens Used: {tokens_used}")

    return labeled_chunks

# Function to extract structured data from labeled chunks
def extract_data(labeled_chunks: List[Dict], topic_definitions: Dict[str, str]) -> dm.ExtractedData:
    start_time = time.time()

    prompt = f"""Extract structured data based on these topic definitions:
    {json.dumps(topic_definitions, indent=2)}"""
    
    response = clientOAI.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(labeled_chunks)}],
        response_format="json"
    )

    tokens_used = response.usage.total_tokens
    elapsed_time = time.time() - start_time
    print(f"[Extract Data] Time: {elapsed_time:.2f}s, Tokens Used: {tokens_used}")
    
    return dm.ExtractedData.parse_raw(response.choices[0].message.content)

# Main function to process a given URL
def process_url(url: str, topic_definitions: Dict[str, str]):
    start_time = time.time()

    data = fetch_and_parse(url)

    #TEST PRINTS
    print(f"\n========== Extracted Text ==========\n\n\n\n {data.text}")
    print("\n========== Extracted Tables ==========\n")
    for table in data.tables:
        print(f"\n{json.dumps(table, indent=2)}:\n")
        if table.data:
            print(tabulate(table.data, headers="keys", tablefmt="grid"))
        else:
            print("No valid data extracted.")
        print("\n" + "="*50)  # Separator for better readability
    print(f"\n========== Extracted Hyperlinks ==========\n\n\n\n {data.hyperlinks}")
    print(f"\n========== Extracted Images ==========\n\n\n\n {data.images}")

    #chunks = chunk(data)
    #structured_data = extract_data(labeled_chunks, topic_definitions)

    total_time = time.time() - start_time
    print(f"Total Process Time: {total_time:.2f}s")

    # Placeholder to save output in database
    
    #return structured_data

# Example topic definitions (can be loaded from a JSON file)
topic_definitions = {
    "Company Name": "The official name of the mining company.",
    "Mine Name": "The name of the mine site.",
    "Production Figures": "Annual production figures in tons.",
    "Cost Figures": "Operational and capital costs.",
    "M&A Activity": "Mergers, acquisitions, and investment activity."
}

# Example usage
if __name__ == "__main__":
    test_url = "https://evolutionmining.com.au/storage/2024/02/2680687-Annual-Mineral-Resources-and-Ore-Reserves-Statement.pdf"
    process_url(test_url, topic_definitions)