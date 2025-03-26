import os
import hashlib
import logging
import json
import re
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import requests
from bs4 import BeautifulSoup
import PyPDF2
import fitz  # PyMuPDF
import pandas as pd
import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from tqdm import tqdm
import concurrent.futures
from pydantic import BaseModel, Field
import openai

# Import Prefect for workflow orchestration
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote
from datetime import timedelta

# ------------------------------
# Data Models for Structured Data
# ------------------------------

class Grade(BaseModel):
    """Grade values for different metals (e.g., Au, Cu, Zn)"""
    values: Dict[str, float]  # Dictionary where keys are metal names and values are grades

class Category(BaseModel):
    """Represents ore tonnage, contained metal, and grade for a resource or reserve category."""
    ore: Optional[float] = None  # Ore tonnage in metric tonnes
    metal: Optional[float] = None  # Contained metal in metric tonnes
    grade: Optional[Grade] = None  # Grade per metal

class Method(BaseModel):
    """Mining method (e.g., Open Pit, Underground, Stockpile)."""
    method: Optional[str] = None  # Mining method name
    measured: Optional[Category] = None
    indicated: Optional[Category] = None
    inferred: Optional[Category] = None
    proven: Optional[Category] = None
    probable: Optional[Category] = None

class Deposit(BaseModel):
    """Deposit within a mine site, normally proper names or codes."""
    deposit: Optional[str] = None  # Name of the deposit
    methods: List[Method] = Field(default_factory=list)  # Mining methods

class MineSite(BaseModel):
    """Mine site containing different deposits."""
    mine_site: str  # Name of the mine site
    deposits: List[Deposit] = Field(default_factory=list)  # Deposits

class ReservesAndResources(BaseModel):
    """Top-level model to hold both resources and reserves."""
    resources: List[MineSite] = Field(default_factory=list)
    reserves: List[MineSite] = Field(default_factory=list)

class ExtractedData(BaseModel):
    """Structured data extracted from a document."""
    document_name: str
    document_date: Optional[str] = None
    document_author: Optional[str] = None
    topics: Dict[str, str]

class LabeledChunk(BaseModel):
    """Labeled chunk of text for better retrieval."""
    chunk_index: int
    labels: List[str]

class TableSchema(BaseModel):
    """Table extracted from a document."""
    title: str
    data: List[Dict[str, Union[str, float, int]]] = Field(default_factory=list)

class ImageSchema(BaseModel):
    """Image extracted from a document."""
    url: str
    description: str
    mine_site: str
    category: str  # e.g., "processing plant", "open pit", etc.

class ParsedDocument(BaseModel):
    """Fully parsed document with all elements."""
    text: str
    tables: List[Any] = Field(default_factory=list)  # Will store table data
    hyperlinks: List[str] = Field(default_factory=list)
    images: List[ImageSchema] = Field(default_factory=list)

class Document(BaseModel):
    """Document with content and metadata for vector storage."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ------------------------------
# Web Scraper Component
# ------------------------------

@task(
    name="discover_pdf_links",
    description="Crawl a website to discover PDF links",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6)
)
def discover_pdf_links(base_url: str, max_pages: int = 10) -> List[Dict[str, str]]:
    """
    Crawl the website to discover PDF links using breadth-first search.
    
    Args:
        base_url: Base URL to begin crawling from
        max_pages: Maximum number of pages to crawl to limit scope
        
    Returns:
        List of dictionaries with pdf_url, source_page, title, and discovery date
    """
    logger = get_run_logger()
    logger.info(f"Discovering PDF links from {base_url}")
    
    # Set up a session for efficient requests with persistent connections
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +http://www.example.com)"
    })
    
    visited_urls = set()  # Track visited pages to avoid loops
    to_visit = [base_url]  # Queue of pages to visit
    pdf_links = []  # Store discovered PDF links
    
    # Use breadth-first search to crawl the site
    with tqdm(total=max_pages, desc="Crawling pages") as pbar:
        while to_visit and len(visited_urls) < max_pages:
            url = to_visit.pop(0)  # Get next URL to visit
            
            # Skip if already visited
            if url in visited_urls:
                continue
                
            try:
                # Fetch the page with a timeout to avoid hanging
                response = session.get(url, timeout=10)
                visited_urls.add(url)
                pbar.update(1)
                
                # Check if the page was successfully fetched
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status_code}")
                    continue
                    
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links in the page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Handle various URL formats (absolute, relative, etc.)
                    if href.startswith('/'):
                        # Convert root-relative URLs to absolute URLs
                        href = f"{base_url.rstrip('/')}{href}"
                    elif not href.startswith(('http://', 'https://')):
                        # Convert page-relative URLs to absolute URLs
                        href = f"{'/'.join(url.split('/')[:-1])}/{href}"
                        
                    # Check if the link points to a PDF
                    if href.lower().endswith('.pdf'):
                        # Extract title from link text or use filename as fallback
                        title = link.get_text().strip() or os.path.basename(href)
                        pdf_links.append({
                            'pdf_url': href,
                            'source_page': url,
                            'title': title,
                            'discovery_date': datetime.now().isoformat()
                        })
                        logger.info(f"Found PDF: {href}")
                        
                    # Add to crawl queue if it's on the same domain and not yet processed
                    elif href.startswith(base_url) and href not in visited_urls and href not in to_visit:
                        to_visit.append(href)
                        
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                
    logger.info(f"Discovered {len(pdf_links)} PDF links")
    return pdf_links

@task(
    name="download_pdfs",
    description="Download PDFs from discovered links",
    retries=3,
    retry_delay_seconds=30
)
def download_pdfs(pdf_links: List[Dict[str, str]], output_dir: str, max_pdfs: Optional[int] = None) -> Dict[str, Dict]:
    """
    Download PDFs from the discovered links and track their metadata.
    
    Args:
        pdf_links: List of dictionaries with pdf_url keys from discover_pdf_links
        output_dir: Directory to save downloaded PDFs
        max_pdfs: Maximum number of PDFs to download (None for all)
        
    Returns:
        Dictionary mapping filenames to their metadata including local paths
    """
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up a session for efficient requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +http://www.example.com)"
    })
    
    # Limit the number of PDFs to download if specified
    if max_pdfs:
        pdf_links = pdf_links[:max_pdfs]
        
    downloaded_files = {}
    
    # Show progress bar for downloads
    with tqdm(total=len(pdf_links), desc="Downloading PDFs") as pbar:
        for pdf_info in pdf_links:
            pdf_url = pdf_info['pdf_url']
            # Sanitize filename to ensure it's safe for the filesystem
            filename = sanitize_filename(os.path.basename(pdf_url))
            local_path = os.path.join(output_dir, filename)
            
            # Skip downloading if file already exists - just update metadata
            if os.path.exists(local_path):
                # Hash the existing file for change detection
                file_hash = hash_file(local_path)
                downloaded_files[filename] = {
                    'path': local_path,
                    'url': pdf_url,
                    'source_page': pdf_info['source_page'],
                    'title': pdf_info['title'],
                    'hash': file_hash,
                    'download_date': datetime.now().isoformat(),
                    'already_existed': True
                }
                pbar.update(1)
                continue
                
            try:
                # Stream the download to handle large files efficiently
                response = session.get(pdf_url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Save the file in chunks to manage memory usage
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    # Hash the downloaded file for change detection
                    file_hash = hash_file(local_path)
                    downloaded_files[filename] = {
                        'path': local_path,
                        'url': pdf_url,
                        'source_page': pdf_info['source_page'],
                        'title': pdf_info['title'],
                        'hash': file_hash,
                        'download_date': datetime.now().isoformat(),
                        'already_existed': False
                    }
                    logger.info(f"Downloaded {pdf_url} to {local_path}")
                else:
                    logger.warning(f"Failed to download {pdf_url}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error downloading {pdf_url}: {str(e)}")
                
            pbar.update(1)
    
    # Save metadata for future reference
    metadata_path = os.path.join(output_dir, 'metadata.json')
    pd.DataFrame.from_dict(downloaded_files, orient='index').to_json(
        metadata_path, orient='index', indent=2
    )
    
    logger.info(f"Downloaded {len(downloaded_files)} PDFs")
    return downloaded_files

@task(
    name="detect_changes",
    description="Detect changes in previously downloaded PDFs",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def detect_changes(download_dir: str) -> List[str]:
    """
    Detect changes in already downloaded PDFs by comparing hashes.
    This enables incremental updates of the pipeline.
    
    Args:
        download_dir: Directory containing downloaded PDFs
        
    Returns:
        List of filenames for PDFs that have changed since last run
    """
    logger = get_run_logger()
    changed_files = []
    metadata_path = os.path.join(download_dir, 'metadata.json')
    
    # Check if metadata file exists from previous runs
    if not os.path.exists(metadata_path):
        logger.warning("No metadata file found. Cannot detect changes.")
        return changed_files
        
    # Load previous metadata
    old_metadata = pd.read_json(metadata_path, orient='index').to_dict(orient='index')
    
    # Check each file for changes
    for filename, metadata in old_metadata.items():
        filepath = metadata['path']
        old_hash = metadata['hash']
        
        if os.path.exists(filepath):
            # Compare current hash with stored hash
            new_hash = hash_file(filepath)
            if new_hash != old_hash:
                changed_files.append(filename)
                logger.info(f"Detected changes in {filename}")
                
    return changed_files

# ------------------------------
# PDF Processing Component
# ------------------------------

@task(
    name="process_pdf",
    description="Process a PDF to extract text and structured data",
    retries=2,
    retry_delay_seconds=30
)
def process_pdf(filename: str, filepath: str, metadata: Dict, output_dir: str, topic_definitions: Dict[str, str]) -> Dict:
    """
    Extract structured data from a PDF using multiple methods.
    Uses both PyPDF2 and PyMuPDF for redundancy, as different PDF libraries
    handle different document structures with varying success.
    
    Args:
        filename: Name of the PDF file
        filepath: Path to the PDF file
        metadata: Metadata of the PDF
        output_dir: Directory to save processed data
        topic_definitions: Dictionary of topic definitions for classification
        
    Returns:
        Dictionary with extracted text, structured data, and metadata
    """
    logger = get_run_logger()
    
    # Initialize topic classifier for PDF content
    vectorizer, vectors, topic_names = initialize_topic_classifier(topic_definitions)
    
    result = {
        'original_metadata': metadata,
        'hash': metadata['hash'],
        'extraction_date': datetime.now().isoformat(),
        'extraction_methods': {}
    }
    
    # Method 1: Extract text using PyPDF2 (more reliable for some PDFs)
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            result['num_pages'] = num_pages
            
            text = ""
            for i in range(num_pages):
                page = reader.pages[i]
                text += page.extract_text() + "\n\n"
                
            result['extraction_methods']['pypdf2'] = {
                'success': True,
                'text_length': len(text)
            }
            result['text_pypdf2'] = text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed for {filename}: {str(e)}")
        result['extraction_methods']['pypdf2'] = {
            'success': False,
            'error': str(e)
        }
        
    # Method 2: Extract text and elements using PyMuPDF (better for complex PDFs)
    try:
        doc = fitz.open(filepath)
        num_pages = doc.page_count
        result['num_pages'] = num_pages
        
        text = ""
        hyperlinks = []  # Store all hyperlinks found in the document
        images = []      # Store all images found in the document
        
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n\n"
            
            # Extract hyperlinks from the page
            links = page.get_links()
            for link in links:
                if 'uri' in link:
                    hyperlinks.append(link['uri'])
            
            # Extract images from the page
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    img_filename = f"{os.path.splitext(filename)[0]}_page{page_num+1}_img{img_index+1}.png"
                    img_path = os.path.join(output_dir, "images", img_filename)
                    
                    # Create images directory if it doesn't exist
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    
                    # Save image if it's RGB
                    if pix.n - pix.alpha < 4:  # if it's RGB
                        pix.save(img_path)
                    else:  # CMYK: convert to RGB first
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(img_path)
                        pix_rgb = None
                    pix = None
                    
                    # Create ImageSchema with placeholder values (would be enhanced in production)
                    image_schema = ImageSchema(
                        url=img_path,
                        description=f"Image from page {page_num+1}",
                        mine_site="Unknown",  # This would need to be derived from context
                        category="Unknown"    # This would need to be derived from content analysis
                    )
                    images.append(image_schema)
                except Exception as e:
                    logger.warning(f"Error extracting image from {filename}, page {page_num+1}: {str(e)}")
            
        # Extract document metadata (author, creation date, etc.)
        pdf_metadata = doc.metadata
        result['document_metadata'] = {k: v for k, v in pdf_metadata.items()}
        
        # Extract table of contents if available
        toc = doc.get_toc()
        if toc:
            result['table_of_contents'] = toc
            
        result['extraction_methods']['pymupdf'] = {
            'success': True,
            'text_length': len(text)
        }
        result['text_pymupdf'] = text
        result['hyperlinks'] = hyperlinks
        result['images'] = [img.dict() for img in images]  # Convert Pydantic models to dict
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {filename}: {str(e)}")
        result['extraction_methods']['pymupdf'] = {
            'success': False,
            'error': str(e)
        }
        
    # Method 3: Extract tables using Camelot (specialized for table extraction)
    try:
        tables = camelot.read_pdf(filepath)
        result['extraction_methods']['camelot'] = {
            'success': True,
            'num_tables': len(tables)
        }
        result['tables'] = tables  # This is a TableList object from camelot
        
        # Convert tables to TableSchema objects for standardized access
        table_schemas = []
        for i, table in enumerate(tables):
            df = table.df
            # Convert DataFrame to list of dictionaries for JSON serialization
            records = df.to_dict('records')
            # Create TableSchema using our Pydantic model
            table_schema = TableSchema(
                title=f"Table {i+1}",
                data=records
            )
            table_schemas.append(table_schema.dict())
        result['table_schemas'] = table_schemas
        
    except Exception as e:
        logger.warning(f"Camelot table extraction failed for {filename}: {str(e)}")
        result['extraction_methods']['camelot'] = {
            'success': False,
            'error': str(e)
        }
        
    # Determine best text extraction method based on success and completeness
    if result['extraction_methods'].get('pymupdf', {}).get('success', False):
        result['text'] = result['text_pymupdf']
        result['extraction_method_used'] = 'pymupdf'
    elif result['extraction_methods'].get('pypdf2', {}).get('success', False):
        result['text'] = result['text_pypdf2']
        result['extraction_method_used'] = 'pypdf2'
    else:
        result['text'] = ""
        result['extraction_method_used'] = 'none'
        
    # Post-process text to clean up extraction artifacts
    result['text'] = clean_text(result['text'])
    
    # Extract logical sections from the document
    result['sections'] = extract_sections(result['text'])
    
    # Classify text into topics using TF-IDF similarity
    result['topics'] = classify_text(result['text'], vectorizer, vectors, topic_names, topic_definitions)
    
    # Create structured ExtractedData model for standardized access
    document_name = metadata.get('title', filename)
    document_date = result['document_metadata'].get('creationDate', None)
    document_author = result['document_metadata'].get('author', None)
    
    # Create ExtractedData model to hold the core metadata and topics
    extracted_data = ExtractedData(
        document_name=document_name,
        document_date=document_date,
        document_author=document_author,
        topics=result['topics']
    )
    result['structured_data'] = extracted_data.dict()
    
    # Create ParsedDocument model to hold the full document content
    try:
        parsed_document = ParsedDocument(
            text=result['text'],
            tables=result.get('tables', []),
            hyperlinks=result.get('hyperlinks', []),
            images=[ImageSchema(**img) for img in result.get('images', [])]
        )
        result['parsed_document'] = parsed_document.dict()
    except Exception as e:
        logger.error(f"Error creating ParsedDocument for {filename}: {str(e)}")
    
    # Save processed text to file for manual inspection if needed
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])
        
    # Save structured data as JSON for external use
    structured_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_structured.json")
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump(result['structured_data'], f, indent=2)
        
    result['processed_path'] = output_path
    result['structured_path'] = structured_path
    return result

@task(
    name="process_pdfs",
    description="Process all PDFs in batch",
    retries=1,
    retry_delay_seconds=60
)
def process_pdfs(pdf_metadata: Dict[str, Dict], 
                input_dir: str, 
                output_dir: str, 
                topic_definitions: Dict[str, str],
                reprocess_all: bool = False) -> Dict[str, Dict]:
    """
    Process all PDFs in the input directory.
    
    Args:
        pdf_metadata: Metadata of PDFs to process
        input_dir: Input directory containing PDFs
        output_dir: Output directory for processed files
        topic_definitions: Dictionary of topic definitions for classification
        reprocess_all: Whether to reprocess all PDFs or just new ones
        
    Returns:
        Dictionary with processing results
    """
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_path = os.path.join(output_dir, 'processed_metadata.json')
    
    # Load existing processed metadata if available to avoid duplicate work
    if os.path.exists(processed_path) and not reprocess_all:
        processed_metadata = pd.read_json(processed_path, orient='index').to_dict(orient='index')
    else:
        processed_metadata = {}
        
    # Determine which files to process - only process new or changed PDFs unless forced
    to_process = []
    for filename, metadata in pdf_metadata.items():
        filepath = metadata['path']
        
        # Skip if already processed and unchanged (using hash comparison)
        if (filename in processed_metadata and 
            metadata['hash'] == processed_metadata[filename].get('hash') and
            not reprocess_all):
            continue
            
        to_process.append((filename, filepath, metadata))
        
    logger.info(f"Processing {len(to_process)} PDFs")
    
    # Process PDFs in parallel using ThreadPoolExecutor for better performance
    with tqdm(total=len(to_process), desc="Processing PDFs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(process_pdf, filename, filepath, metadata, output_dir, topic_definitions): 
                filename for filename, filepath, metadata in to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    processed_metadata[filename] = result
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                pbar.update(1)
                
    # Save processed metadata for future runs
    pd.DataFrame.from_dict(processed_metadata, orient='index').to_json(
        processed_path, orient='index', indent=2
    )
    
    logger.info(f"Processed {len(to_process)} PDFs")
    return processed_metadata

# ------------------------------
# Document Chunking Component
# ------------------------------

@task(
    name="chunk_document",
    description="Chunk a document into semantic units for better retrieval",
    retries=2,
    retry_delay_seconds=15
)
def chunk_document(filename: str, metadata: Dict, 
                  output_dir: str, 
                  topic_definitions: Dict[str, str]) -> List[Document]:
    """
    Chunk a single document and label each chunk with relevant topics.
    Preserves document structure when possible by using sections.
    
    Args:
        filename: Name of the document
        metadata: Metadata of the document
        output_dir: Directory to save chunked results
        topic_definitions: Dictionary of topic definitions for labeling
        
    Returns:
        List of chunked and labeled documents
    """
    logger = get_run_logger()
    
    # Initialize topic classifier for chunk labeling
    vectorizer, vectors, topic_names = initialize_topic_classifier(topic_definitions)
    
    text = metadata['text']
    title = metadata['original_metadata']['title']
    source_url = metadata['original_metadata']['url']
    
    # Get structured data if available for enriching chunk metadata
    structured_data = metadata.get('structured_data', {})
    
    # Initialize the text splitter for chunking 
    chunk_size = 1000
    chunk_overlap = 200
    
    # First try semantic chunking using sections to preserve document structure
    chunks = []
    if 'sections' in metadata and metadata['sections']:
        for i, section in enumerate(metadata['sections']):
            section_text = section['text']
            section_title = section['title']
            
            # Skip very short sections (likely headers without content)
            if len(section_text) < 100:
                continue
                
            # Split section text into chunks of appropriate size with overlap
            section_chunks = split_text(
                text=section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata={
                    'source': filename,
                    'title': title,
                    'section': section_title,
                    'section_idx': i,
                    'url': source_url,
                    'chunk_type': 'section'
                }
            )
            chunks.extend(section_chunks)
    
    # If no sections or very few chunks, fall back to whole document chunking
    if len(chunks) < 3:
        chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={
                'source': filename,
                'title': title,
                'url': source_url,
                'chunk_type': 'document'
            }
        )
        
    # Add chunk index, labels, and enhanced metadata
    labeled_chunks = []
    for i, chunk in enumerate(chunks):
        # Label the chunk with relevant topics
        labels = label_chunk(chunk.page_content, vectorizer, vectors, topic_names)
        
        # Create LabeledChunk model for persistent storage
        labeled_chunk = LabeledChunk(
            chunk_index=i,
            labels=labels
        )
        
        # Add metadata to chunk for improved retrieval
        chunk.metadata['chunk_idx'] = i
        chunk.metadata['total_chunks'] = len(chunks)
        chunk.metadata['labels'] = labels
        
        # Add structured data references if available
        if structured_data:
            chunk.metadata['document_name'] = structured_data.get('document_name', '')
            chunk.metadata['document_date'] = structured_data.get('document_date', '')
            chunk.metadata['document_author'] = structured_data.get('document_author', '')
            
            # Add topic data if this chunk matches any of the classified topics
            # This helps identify the "primary topic" of each chunk
            if 'topics' in metadata:
                for topic, topic_text in metadata['topics'].items():
                    if topic_text in chunk.page_content:
                        chunk.metadata['primary_topic'] = topic
                        break
        
        # Extract likely keywords from the chunk for additional retrieval signals
        keywords = extract_keywords(chunk.page_content)
        chunk.metadata['keywords'] = keywords
        
        # Store the labeled chunk
        labeled_chunks.append(chunk)
        
        # Save labeled chunk data as JSON for external use
        labeled_chunk_path = os.path.join(
            output_dir, 
            f"{os.path.splitext(filename)[0]}_chunk_{i}_labels.json"
        )
        with open(labeled_chunk_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_chunk.dict(), f, indent=2)
        
    return labeled_chunks

@task(
    name="chunk_documents",
    description="Chunk all documents in batch",
    retries=1,
    retry_delay_seconds=30
)
def chunk_documents(processed_metadata: Dict[str, Dict], 
                   chunked_dir: str,
                   topic_definitions: Dict[str, str],
                   rechunk_all: bool = False) -> List[Document]:
    """
    Chunk all processed documents and label the chunks.
    Handles incremental processing to avoid rechunking unchanged documents.
    
    Args:
        processed_metadata: Metadata of processed documents
        chunked_dir: Directory to save chunked documents
        topic_definitions: Dictionary of topic definitions for labeling
        rechunk_all: Whether to rechunk all documents or just new ones
        
    Returns:
        List of chunked and labeled documents
    """
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(chunked_dir):
        os.makedirs(chunked_dir)
    
    chunked_path = os.path.join(chunked_dir, 'chunked_metadata.json')
    
    # Load existing chunked metadata if available
    if os.path.exists(chunked_path) and not rechunk_all:
        chunked_metadata = pd.read_json(chunked_path, orient='index').to_dict(orient='index')
    else:
        chunked_metadata = {}
        
    # Determine which files to chunk - only process new or changed documents
    to_chunk = []
    for filename, metadata in processed_metadata.items():
        # Skip if already chunked and unchanged (using hash comparison)
        if (filename in chunked_metadata and 
            metadata['hash'] == chunked_metadata[filename].get('hash') and
            not rechunk_all):
            continue
            
        to_chunk.append((filename, metadata))
        
    logger.info(f"Chunking {len(to_chunk)} documents")
    
    all_chunks = []
    
    with tqdm(total=len(to_chunk), desc="Chunking documents") as pbar:
        for filename, metadata in to_chunk:
            try:
                chunks = chunk_document(filename, metadata, chunked_dir, topic_definitions)
                all_chunks.extend(chunks)
                
                # Update chunked metadata to avoid reprocessing unchanged documents
                chunked_metadata[filename] = {
                    'hash': metadata['hash'],
                    'chunking_date': datetime.now().isoformat(),
                    'num_chunks': len(chunks)
                }
            except Exception as e:
                logger.error(f"Error chunking {filename}: {str(e)}")
                
            pbar.update(1)
            
    # Save chunked metadata for future runs
    pd.DataFrame.from_dict(chunked_metadata, orient='index').to_json(
        chunked_path, orient='index', indent=2
    )
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(to_chunk)} documents")
    return all_chunks

# ------------------------------
# Vector Database Component
# ------------------------------

@task(
    name="initialize_vector_db",
    description="Initialize or load the vector database",
    retries=2,
    retry_delay_seconds=15
)
def initialize_vector_db(db_dir: str) -> chromadb.PersistentClient:
    """
    Initialize or load the vector database.
    
    Args:
        db_dir: Directory to store vector database
        
    Returns:
        Initialized vector database client
    """
    logger = get_run_logger()
    
    # Create directory if it doesn't exist
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_dir)
    
    # Get or create the collection for documents
    try:
        collection = client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        logger.info(f"Initialized vector database at {db_dir}")
        return client
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        raise

@task(
    name="embed_and_index_documents",
    description="Create embeddings and index documents in vector database",
    retries=2,
    retry_delay_seconds=30
)
def embed_and_index_documents(documents: List[Document], 
                             client: chromadb.PersistentClient, 
                             db_dir: str,
                             batch_size: int = 32) -> None:
    """
    Create embeddings and index documents in the vector database.
    
    Args:
        documents: List of documents to index
        client: ChromaDB client
        db_dir: Directory where vector database is stored
        batch_size: Batch size for indexing to manage API limits
    """
    logger = get_run_logger()
    logger.info(f"Indexing {len(documents)} documents in vector database")
    
    # Get collection
    collection = client.get_collection("rag_documents")
    
    # Path for storing complex metadata separately
    metadata_store_path = os.path.join(db_dir, "metadata_store.json")
    
    # Load existing metadata store if available
    if os.path.exists(metadata_store_path):
        try:
            with open(metadata_store_path, 'r', encoding='utf-8') as f:
                metadata_store = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata store: {str(e)}")
            metadata_store = {}
    else:
        metadata_store = {}
    
    # Process in batches to manage API rate limits and memory
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    with tqdm(total=len(documents), desc="Indexing documents") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            try:
                # Prepare batch data for ChromaDB
                ids = [f"doc_{i+j}" for j in range(len(batch))]
                texts = [doc.page_content for doc in batch]
                
                # Prepare metadata, filtering out complex structures
                metadatas = []
                for j, doc in enumerate(batch):
                    doc_id = ids[j]
                    
                    # Simplified metadata for ChromaDB
                    simple_metadata = {}
                    enhanced_metadata = {}
                    
                    for key, value in doc.metadata.items():
                        # Store simple types directly in ChromaDB
                        if isinstance(value, (str, int, float, bool)) or (
                            isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value)
                        ):
                            simple_metadata[key] = value
                        # Store complex types in separate metadata store
                        elif isinstance(value, (list, dict)) or key in ['labels', 'keywords']:
                            enhanced_metadata[key] = value
                    
                    metadatas.append(simple_metadata)
                    
                    # Store complex metadata separately
                    if enhanced_metadata:
                        metadata_store[doc_id] = enhanced_metadata
                
                # Get embeddings from OpenAI API
                embeddings = get_embeddings_batch(texts)
                
                # Add to ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                # Sleep to avoid API rate limits
                if i + batch_size < len(documents):
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error indexing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                
            pbar.update(len(batch))
    
    # Save metadata store
    try:
        with open(metadata_store_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata store: {str(e)}")
    
    logger.info(f"Indexed {len(documents)} documents")

@task(
    name="semantic_search",
    description="Perform semantic search in the vector database"
)
def semantic_search(query: str, 
                   client: chromadb.PersistentClient, 
                   db_dir: str,
                   k: int = 5, 
                   filter_labels: List[str] = None) -> List[Document]:
    """
    Perform semantic search in the vector database with optional filtering.
    
    Args:
        query: Query text
        client: ChromaDB client
        db_dir: Directory where vector database is stored
        k: Number of results to return
        filter_labels: Optional list of labels to filter by
        
    Returns:
        List of similar documents
    """
    logger = get_run_logger()
    
    # Get collection
    collection = client.get_collection("rag_documents")
    
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Path for metadata store
    metadata_store_path = os.path.join(db_dir, "metadata_store.json")
    
    # Load metadata store
    if os.path.exists(metadata_store_path):
        try:
            with open(metadata_store_path, 'r', encoding='utf-8') as f:
                metadata_store = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata store: {str(e)}")
            metadata_store = {}
    else:
        metadata_store = {}
    
    try:
        # Basic similarity search without filtering
        if not filter_labels:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            # First get more results than needed to allow for filtering
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k*2,
                include=["documents", "metadatas", "distances", "ids"]
            )
            
            # Filter results by labels
            filtered_indices = []
            for i, doc_id in enumerate(results['ids'][0]):
                # Get enhanced metadata from store
                enhanced_metadata = metadata_store.get(doc_id, {})
                doc_labels = enhanced_metadata.get('labels', [])
                
                # Check if any of the required labels are present
                if any(label in doc_labels for label in filter_labels):
                    filtered_indices.append(i)
                    
                if len(filtered_indices) >= k:
                    break
            
            # Filter results to only keep matches
            if filtered_indices:
                results = {
                    'ids': [[results['ids'][0][i] for i in filtered_indices]],
                    'documents': [[results['documents'][0][i] for i in filtered_indices]],
                    'metadatas': [[results['metadatas'][0][i] for i in filtered_indices]],
                    'distances': [[results['distances'][0][i] for i in filtered_indices]]
                }
            else:
                # If no matches, limit to k results
                results = {
                    'ids': [results['ids'][0][:k]],
                    'documents': [results['documents'][0][:k]],
                    'metadatas': [results['metadatas'][0][:k]],
                    'distances': [results['distances'][0][:k]]
                }
        
        # Convert results to Document objects
        docs = []
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i] if 'ids' in results else f"result_{i}"
            doc_text = results['documents'][0][i]
            doc_metadata = results['metadatas'][0][i].copy() if results['metadatas'][0][i] else {}
            
            # Add enhanced metadata from store
            if doc_id in metadata_store:
                for key, value in metadata_store[doc_id].items():
                    doc_metadata[key] = value
            
            # Create Document object
            doc = Document(
                page_content=doc_text,
                metadata=doc_metadata
            )
            docs.append(doc)
        
        return docs
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return []

# ------------------------------
# Query Engine Component
# ------------------------------

@task(
    name="classify_query",
    description="Classify query into relevant topics"
)
def classify_query(query: str, topic_definitions: Dict[str, str]) -> List[str]:
    """
    Classify the query into relevant topics using semantic similarity.
    
    Args:
        query: Query text
        topic_definitions: Dictionary of topic definitions
        
    Returns:
        List of relevant topic labels
    """
    # Initialize topic classifier
    vectorizer, vectors, topic_names = initialize_topic_classifier(topic_definitions)
    
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity with each topic definition
    similarities = cosine_similarity(query_vector, vectors).flatten()
    
    # Get topics with similarity above threshold
    relevant_topics = []
    for i, score in enumerate(similarities):
        if score > 0.1:  # Threshold for relevance - can be tuned
            relevant_topics.append(topic_names[i])
            
    return relevant_topics

@task(
    name="extract_structured_data",
    description="Extract structured data from context documents"
)
def extract_structured_data(docs: List[Document], topics: List[str]) -> Dict[str, Any]:
    """
    Extract structured data from context documents based on query topics.
    
    Args:
        docs: List of context documents
        topics: Query topics
        
    Returns:
        Extracted structured data organized by topic
    """
    structured_data = {}
    
    # Extract company names if that's a relevant topic
    if "Company Name" in topics:
        companies = set()
        for doc in docs:
            if 'structured_data' in doc.metadata:
                if 'document_name' in doc.metadata['structured_data']:
                    companies.add(doc.metadata['structured_data']['document_name'])
        structured_data['companies'] = list(companies)
        
    # Extract mine names if that's a relevant topic
    if "Mine Name" in topics:
        mines = set()
        for doc in docs:
            if 'mine_site' in doc.metadata:
                mines.add(doc.metadata['mine_site'])
        structured_data['mines'] = list(mines)
        
    # Extract resources and reserves data
    resources = []
    reserves = []
    for doc in docs:
        if 'structured_data' in doc.metadata:
            if 'resources' in doc.metadata['structured_data']:
                resources.extend(doc.metadata['structured_data']['resources'])
            if 'reserves' in doc.metadata['structured_data']:
                reserves.extend(doc.metadata['structured_data']['reserves'])
                
    if resources:
        structured_data['resources'] = resources
    if reserves:
        structured_data['reserves'] = reserves
        
    return structured_data

@task(
    name="generate_answer",
    description="Generate answer from context documents using LLM"
)
def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    Generate an answer for the query using retrieved context documents and the LLM.
    
    Args:
        query: The query text
        context_docs: The retrieved context documents
        
    Returns:
        Generated answer
    """
    logger = get_run_logger()
    
    # Prepare context by joining document content
    context = ""
    for i, doc in enumerate(context_docs):
        # Add document content
        context += f"\n\nDocument {i+1}:\n{doc.page_content}"
        
        # Add structured data if available
        if 'structured_data' in doc.metadata:
            structured_str = json.dumps(doc.metadata['structured_data'], indent=2)
            context += f"\n\nStructured Data:\n{structured_str}"
    
    # Prepare prompt for the LLM
    prompt = f"""
    Answer the following question based on the provided context. If the answer is not
    in the context, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    try:
        # Call OpenAI API to generate answer
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        # Extract answer from response
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"An error occurred while generating the answer: {str(e)}"

@task(
    name="process_query",
    description="Process a user query and generate an answer"
)
def process_query(query: str, 
                 client: chromadb.PersistentClient,
                 db_dir: str,
                 topic_definitions: Dict[str, str],
                 k: int = 5,
                 use_structured_data: bool = True) -> Dict[str, Any]:
    """
    Process a query using the RAG system with structured data.
    
    Args:
        query: Query text
        client: ChromaDB client
        db_dir: Directory where vector database is stored
        topic_definitions: Dictionary of topic definitions for classification
        k: Number of context documents to retrieve
        use_structured_data: Whether to use structured data in response
        
    Returns:
        Dictionary with query results, sources, and structured data
    """
    logger = get_run_logger()
    logger.info(f"Processing query: {query}")
    
    # Step 1: Classify query into topics for targeted retrieval
    query_topics = classify_query(query, topic_definitions)
    logger.info(f"Query topics: {query_topics}")
    
    # Step 2: Get context documents, filtered by topic if available
    if query_topics and use_structured_data:
        context_docs = semantic_search(
            query, client, db_dir, k=k, filter_labels=query_topics
        )
    else:
        context_docs = semantic_search(query, client, db_dir, k=k)
    
    # Handle case where no relevant documents are found
    if not context_docs:
        return {
            'query': query,
            'answer': "I couldn't find relevant information to answer this query.",
            'sources': [],
            'topics': query_topics
        }
        
    # Step 3: Generate answer with LLM using retrieved context
    answer = generate_answer(query, context_docs)
    
    # Step 4: Extract source information with structured data for attribution
    sources = []
    for doc in context_docs:
        source_info = {
            'content': doc.page_content[:200] + "...",  # Preview of content
            'metadata': {}
        }
        
        # Include most important metadata for source attribution
        for key in ['source', 'title', 'labels', 'section', 'document_name']:
            if key in doc.metadata:
                source_info['metadata'][key] = doc.metadata[key]
                
        sources.append(source_info)
        
    # Step 5: Extract structured data for enhanced response
    structured_data = extract_structured_data(context_docs, query_topics)
        
    return {
        'query': query,
        'answer': answer,
        'sources': sources,
        'topics': query_topics,
        'structured_data': structured_data
    }

# ------------------------------
# Main Pipeline Flow
# ------------------------------

@flow(
    name="rag_pipeline",
    description="Full RAG pipeline from scraping to indexing",
    log_prints=True
)
def rag_pipeline(base_url: str, config: Dict = None):
    """
    Run the full RAG pipeline from scraping to indexing.
    
    Args:
        base_url: Base URL to scrape for PDF documents
        config: Configuration dictionary (optional)
    """
    # Initialize configuration with defaults
    if config is None:
        config = {}
    
    config.setdefault('download_dir', 'downloaded_pdfs')      # Directory for downloaded PDFs
    config.setdefault('processed_dir', 'processed_pdfs')      # Directory for processed text and data
    config.setdefault('chunked_dir', 'chunked_docs')          # Directory for chunked documents
    config.setdefault('vector_db_dir', 'vector_db')           # Directory for vector database
    config.setdefault('max_pdfs', 50)                         # Maximum PDFs to process
    config.setdefault('max_pages', 10)                        # Maximum pages to crawl
    config.setdefault('reprocess', False)                     # Whether to reprocess all documents
    
    # Topic definitions for classification
    topic_definitions = {
        "Company Name": "The official name of the mining company.",
        "Mine Name": "The name of the mine site.",
        "Production Figures": "Annual production figures in tons.",
        "Cost Figures": "Operational and capital costs.",
        "M&A Activity": "Mergers, acquisitions, and investment activity."
    }
    
    print(f"Starting RAG pipeline for {base_url}")
    
    # Step 1: Scrape website and download PDFs
    print("Step 1: Discovering and downloading PDFs")
    pdf_links = discover_pdf_links(base_url, config['max_pages'])
    pdf_metadata = download_pdfs(pdf_links, config['download_dir'], config['max_pdfs'])
    
    # Step 2: Process PDFs to extract text and structured data
    print("Step 2: Processing PDFs")
    processed_metadata = process_pdfs(
        pdf_metadata, 
        config['download_dir'], 
        config['processed_dir'],
        topic_definitions,
        config['reprocess']
    )
    
    # Step 3: Chunk documents into semantic units
    print("Step 3: Chunking documents")
    chunked_docs = chunk_documents(
        processed_metadata, 
        config['chunked_dir'],
        topic_definitions,
        config['reprocess']
    )
    
    # Step 4: Initialize vector database
    print("Step 4: Initializing vector database")
    client = initialize_vector_db(config['vector_db_dir'])
    
    # Step 5: Index chunks in vector database
    print("Step 5: Indexing documents in vector database")
    embed_and_index_documents(chunked_docs, client, config['vector_db_dir'])
    
    print("RAG pipeline completed successfully")
    return {
        'client': client,
        'db_dir': config['vector_db_dir'],
        'topic_definitions': topic_definitions
    }

@flow(
    name="update_rag_pipeline",
    description="Update RAG pipeline with new or changed documents",
    log_prints=True
)
def update_rag_pipeline(base_url: str, config: Dict = None):
    """
    Update the RAG pipeline with new or changed documents.
    
    Args:
        base_url: Base URL to scrape for PDF documents
        config: Configuration dictionary (optional)
    """
    # Initialize configuration with defaults
    if config is None:
        config = {}
    
    config.setdefault('download_dir', 'downloaded_pdfs')
    config.setdefault('processed_dir', 'processed_pdfs')
    config.setdefault('chunked_dir', 'chunked_docs')
    config.setdefault('vector_db_dir', 'vector_db')
    config.setdefault('max_pdfs', 50)
    config.setdefault('max_pages', 10)
    
    # Topic definitions for classification
    topic_definitions = {
        "Company Name": "The official name of the mining company.",
        "Mine Name": "The name of the mine site.",
        "Production Figures": "Annual production figures in tons.",
        "Cost Figures": "Operational and capital costs.",
        "M&A Activity": "Mergers, acquisitions, and investment activity."
    }
    
    print(f"Updating RAG pipeline for {base_url}")
    
    # Step 1: Detect changes in already downloaded PDFs
    print("Step 1: Detecting changes in existing PDFs")
    changed_files = detect_changes(config['download_dir'])
    
    # Step 2: Discover and download new PDFs
    print("Step 2: Discovering and downloading new PDFs")
    pdf_links = discover_pdf_links(base_url, config['max_pages'])
    pdf_metadata = download_pdfs(pdf_links, config['download_dir'])
    
    # Step 3: Process only new or changed PDFs
    print("Step 3: Processing new or changed PDFs")
    processed_metadata = process_pdfs(
        pdf_metadata, 
        config['download_dir'], 
        config['processed_dir'],
        topic_definitions
    )
    
    # Step 4: Chunk only new or changed documents
    print("Step 4: Chunking new or changed documents")
    chunked_docs = chunk_documents(
        processed_metadata, 
        config['chunked_dir'],
        topic_definitions
    )
    
    # Step 5: Initialize vector database
    print("Step 5: Initializing vector database")
    client = initialize_vector_db(config['vector_db_dir'])
    
    # Step 6: Index only new or changed documents
    if chunked_docs:
        print(f"Step 6: Indexing {len(chunked_docs)} new or changed documents")
        embed_and_index_documents(chunked_docs, client, config['vector_db_dir'])
    else:
        print("Step 6: No new documents to index")
    
    print(f"RAG pipeline update completed: {len(chunked_docs)} new/changed documents")
    return {
        'client': client,
        'db_dir': config['vector_db_dir'],
        'topic_definitions': topic_definitions
    }

@flow(
    name="query_rag_system",
    description="Query the RAG system and get an answer",
    log_prints=True
)
def query_rag_system(query: str, db_dir: str = "vector_db", k: int = 5):
    """
    Query the RAG system and get an answer.
    
    Args:
        query: Query text
        db_dir: Directory where vector database is stored
        k: Number of context documents to retrieve
    """
    # Topic definitions for classification
    topic_definitions = {
        "Company Name": "The official name of the mining company.",
        "Mine Name": "The name of the mine site.",
        "Production Figures": "Annual production figures in tons.",
        "Cost Figures": "Operational and capital costs.",
        "M&A Activity": "Mergers, acquisitions, and investment activity."
    }
    
    print(f"Processing query: {query}")
    
    # Initialize vector database
    client = initialize_vector_db(db_dir)
    
    # Process the query
    result = process_query(
        query=query,
        client=client,
        db_dir=db_dir,
        topic_definitions=topic_definitions,
        k=k
    )
    
    # Print the answer
    print(f"\nAnswer: {result['answer']}")
    
    # Print the sources
    print("\nSources:")
    for i, source in enumerate(result['sources']):
        source_title = source['metadata'].get('title', f"Source {i+1}")
        print(f"{i+1}. {source_title}")
    
    return result

# ------------------------------
# Utility Functions
# ------------------------------

def sanitize_filename(filename: str) -> str:
    """
    Clean filename to be filesystem-safe by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for the filesystem
    """
    # Replace characters that are invalid in filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
        
    # Ensure filename is not too long for the filesystem
    if len(filename) > 255:
        base, ext = os.path.splitext(filename)
        filename = base[:255-len(ext)] + ext
        
    return filename

def hash_file(filepath: str) -> str:
    """
    Generate SHA-256 hash of file for change detection.
    
    Args:
        filepath: Path to the file to hash
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.sha256()
    # Process file in chunks to handle large files efficiently
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing common extraction artifacts.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single one for cleaner paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive whitespace that can occur in PDF extractions
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common OCR/extraction issues
    text = text.replace('l l', 'll')  # Common OCR error
    text = text.replace('|', 'I')     # Pipe character often misrecognized as I
    
    # Remove header/footer patterns like page numbers
    text = re.sub(r'\n\d+\s*\n', '\n', text)
    
    return text.strip()

def extract_sections(text: str) -> List[Dict[str, Any]]:
    """
    Extract logical sections from document text based on header patterns.
    
    Args:
        text: Cleaned document text
        
    Returns:
        List of sections with titles and text
    """
    # Split text into lines to analyze line by line
    lines = text.split('\n')
    sections = []
    current_section = {'title': 'Introduction', 'text': ''}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if the line looks like a header using heuristics:
        # - All uppercase (common for headers)
        # - Ends with colon (often a section marker)
        # - Starts with numbered format like 1.2 (common for numbered sections)
        if (line.isupper() or 
            (line.endswith(':') and len(line) < 100) or
            (re.match(r'^[0-9]+\.[0-9]* ', line))):
            
            # Save previous section if not empty
            if current_section['text'].strip():
                sections.append(current_section)
                
            current_section = {'title': line, 'text': ''}
        else:
            current_section['text'] += line + '\n'
            
    # Add the last section
    if current_section['text'].strip():
        sections.append(current_section)
        
    return sections

def initialize_topic_classifier(topic_definitions: Dict[str, str]):
    """
    Initialize the TF-IDF vectorizer for topic classification.
    
    Args:
        topic_definitions: Dictionary of topic definitions
        
    Returns:
        Tuple of vectorizer, vectors, and topic names
    """
    # Create corpus from topic definitions
    corpus = list(topic_definitions.values())
    
    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(corpus)
    topic_names = list(topic_definitions.keys())
    
    return vectorizer, vectors, topic_names

def classify_text(text: str, vectorizer, vectors, topic_names: List[str], 
                 topic_definitions: Dict[str, str]) -> Dict[str, str]:
    """
    Classify text into predefined topics using TF-IDF similarity.
    
    Args:
        text: The document text to classify
        vectorizer: TF-IDF vectorizer
        vectors: TF-IDF vectors for topics
        topic_names: List of topic names
        topic_definitions: Dictionary of topic definitions
        
    Returns:
        Dictionary mapping topics to relevant text snippets
    """
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    
    # Calculate similarity with each topic definition
    similarities = cosine_similarity(text_vector, vectors).flatten()
    
    # Create dictionary of topic classifications with relevant snippets
    topics = {}
    for i, topic_name in enumerate(topic_names):
        # Only include topics with reasonable similarity score
        if similarities[i] > 0.1:
            # Find most relevant text chunks for this topic
            topic_text = extract_topic_text(text, topic_name, vectorizer, topic_definitions)
            topics[topic_name] = topic_text
        
    return topics

def extract_topic_text(text: str, topic: str, vectorizer, topic_definitions: Dict[str, str]) -> str:
    """
    Extract the most relevant text snippet for a given topic.
    
    Args:
        text: The full document text
        topic: The topic to extract text for
        vectorizer: TF-IDF vectorizer
        topic_definitions: Dictionary of topic definitions
        
    Returns:
        Most relevant text snippet for the topic
    """
    # Split text into paragraphs for analysis
    paragraphs = text.split('\n\n')
    
    # Calculate relevance score for each paragraph
    topic_desc = topic_definitions[topic]
    topic_vector = vectorizer.transform([topic_desc])
    
    paragraph_vectors = vectorizer.transform(paragraphs)
    similarities = cosine_similarity(paragraph_vectors, topic_vector).flatten()
    
    # Get the most relevant paragraph based on similarity score
    if len(similarities) > 0:
        best_idx = np.argmax(similarities)
        return paragraphs[best_idx]
    
    return "No relevant information found."

def split_text(text: str, chunk_size: int, chunk_overlap: int, 
              metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Split text into chunks with overlap for better context preservation.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Amount of overlap between chunks
        metadata: Metadata to attach to each chunk
        
    Returns:
        List of Document objects
    """
    # Use recursive splitting to preserve logical boundaries
    chunks = []
    
    # Define separators in order of precedence
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    # Recursive function to split text
    def _split_text(text, separators_idx=0):
        # If we've reached the end of separators or text is small enough, return it
        if separators_idx >= len(separators) or len(text) <= chunk_size:
            return [text]
        
        separator = separators[separators_idx]
        
        # If separator is not in text, move to next separator
        if separator == "" or separator not in text:
            return _split_text(text, separators_idx + 1)
        
        # Split by current separator
        splits = text.split(separator)
        
        # Initialize result and current chunk
        result = []
        current_chunk = []
        current_length = 0
        
        # Process each split
        for split in splits:
            split_with_sep = split + separator if separator else split
            split_length = len(split_with_sep)
            
            # If adding this split would exceed chunk size, finalize current chunk
            if current_length + split_length > chunk_size and current_chunk:
                result.append(separator.join(current_chunk) if separator else "".join(current_chunk))
                
                # If overlap is enabled, keep some of the last splits for context
                if chunk_overlap > 0:
                    # Determine how many splits to keep for overlap
                    overlap_splits = []
                    overlap_length = 0
                    for split_item in reversed(current_chunk):
                        split_item_with_sep = split_item + separator if separator else split_item
                        if overlap_length + len(split_item_with_sep) <= chunk_overlap:
                            overlap_splits.insert(0, split_item)
                            overlap_length += len(split_item_with_sep)
                        else:
                            break
                    
                    current_chunk = overlap_splits
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add split to current chunk
            current_chunk.append(split)
            current_length += split_length
        
        # Add the final chunk if not empty
        if current_chunk:
            result.append(separator.join(current_chunk) if separator else "".join(current_chunk))
        
        return result
    
    # Split the text
    text_chunks = _split_text(text)
    
    # Create Document objects
    for chunk_text in text_chunks:
        chunk_metadata = metadata.copy() if metadata else {}
        chunks.append(Document(
            page_content=chunk_text,
            metadata=chunk_metadata
        ))
    
    return chunks

def label_chunk(text: str, vectorizer, vectors, topic_names: List[str]) -> List[str]:
    """
    Label a chunk of text with relevant topics using semantic similarity.
    
    Args:
        text: Chunk text to label
        vectorizer: TF-IDF vectorizer
        vectors: TF-IDF vectors for topics
        topic_names: List of topic names
        
    Returns:
        List of topic labels that apply to this chunk
    """
    # Vectorize the chunk text
    chunk_vector = vectorizer.transform([text])
    
    # Calculate similarity with each topic definition
    similarities = cosine_similarity(chunk_vector, vectors).flatten()
    
    # Determine relevant labels based on similarity threshold
    labels = []
    for i, score in enumerate(similarities):
        if score > 0.15:  # Similarity threshold - tunable parameter
            labels.append(topic_names[i])
            
    return labels

def extract_keywords(text: str) -> List[str]:
    """
    Extract important keywords from text for improved search capabilities.
    
    Args:
        text: Text to extract keywords from
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction based on term frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    
    for word in words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
        
    # Filter common stopwords that aren't informative for search
    stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'are', 'not', 'has', 'from'}
    keywords = [word for word, freq in word_freq.items() 
               if freq > 1 and word not in stopwords]
               
    return keywords[:10]  # Return top 10 keywords

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using OpenAI API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        # Return zero embeddings as fallback
        return [[0.0] * 1536 for _ in texts]

def get_embeddings_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """
    Get embeddings for a list of texts in batches to avoid API limits.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for API calls
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = get_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
        
        # Sleep to avoid rate limits
        if i + batch_size < len(texts):
            time.sleep(1)
    
    return all_embeddings

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    # OpenAI API key
    openai.api_key = ""
    
    # Configuration
    config = {
        'download_dir': 'downloaded_pdfs',
        'processed_dir': 'processed_pdfs',
        'chunked_dir': 'chunked_docs',
        'vector_db_dir': 'vector_db',
        'max_pdfs': 20,
        'max_pages': 5,
        'reprocess': False
    }
    
    # Run the pipeline
    result = rag_pipeline("./documents/", config)
    
    # Query
    answer = query_rag_system(
        "What are the key findings in the documents?",
        db_dir=result['db_dir']
    )
    
    print("\nDone!")