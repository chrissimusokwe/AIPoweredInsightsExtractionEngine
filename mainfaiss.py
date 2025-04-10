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
from tqdm import tqdm
import concurrent.futures
from pydantic import BaseModel, Field
from hashlib import sha256

import faiss
import pickle

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Import Prefect for workflow orchestration
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote
from datetime import timedelta

# Add necessary FAISS imports
from pydantic import BaseModel, Field

# Document class definition for compatibility
class Document(BaseModel):
    """Document with content and metadata for vector storage."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
# FAISS Vector Database Component
# ------------------------------

class FAISSVectorStore:
    """FAISS Vector Database wrapper for storing and retrieving document embeddings."""
    
    def __init__(self, db_dir: str, dimension: int = 4096):
        """
        Initialize FAISS vector store.
        
        Args:
            db_dir: Directory to store vector database
            dimension: Dimension of the embedding vectors
        """
        self.db_dir = db_dir
        self.dimension = dimension
        
        # Create directory if it doesn't exist
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.index_file = os.path.join(db_dir, "faiss_index.bin")
        self.documents_file = os.path.join(db_dir, "documents.pkl")
        self.metadata_file = os.path.join(db_dir, "metadata_store.json")
        
        # Initialize or load index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize or load FAISS index and associated data."""
        if os.path.exists(self.index_file):
            # Load existing index
            self.index = faiss.read_index(self.index_file)
            
            # Load documents
            with open(self.documents_file, 'rb') as f:
                self.documents = pickle.load(f)
                
            # Load metadata store if it exists
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
            else:
                self.metadata_store = {}
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
            self.documents = []  # List to store document texts
            self.metadata_store = {}  # Dict to store document metadata
    
    def save(self):
        """Save the index and associated data to disk."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)
        
        # Save documents
        with open(self.documents_file, 'wb') as f:
            pickle.dump(self.documents, f)
            
        # Save metadata store
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2)
    
    def count(self):
        """Return the number of documents in the index."""
        return len(self.documents)
    
    def add(self, documents: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict[str, Any]], ids: List[str] = None):
        """
        Add documents and their embeddings to the index.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not documents or not embeddings:
            return
            
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i+len(self.documents)}" for i in range(len(documents))]
            
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Store documents and their metadata
        for i, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
            # Store document
            self.documents.append({
                'id': doc_id,
                'text': document,
                'metadata': metadata
            })
            
            # Store enhanced metadata separately
            enhanced_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    try:
                        # Try to parse JSON strings back to Python objects
                        enhanced_metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        enhanced_metadata[key] = value
                else:
                    enhanced_metadata[key] = value
                    
            self.metadata_store[doc_id] = enhanced_metadata
            
        # Save the updated index and data
        self.save()
    
    def search(self, embedding: List[float], k: int = 5, filter_labels: List[str] = None) -> Dict:
        """
        Search the index for documents similar to the query embedding.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter_labels: Optional list of labels to filter results by
            
        Returns:
            Dictionary with search results
        """
        if len(self.documents) == 0:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
        # Convert embedding to numpy array and normalize
        query_embedding = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Retrieve more results initially if filtering is needed
        num_results = k * 3 if filter_labels else k
        num_results = min(num_results, len(self.documents))
        
        # Search the index
        distances, indices = self.index.search(query_embedding, num_results)
        
        # Convert distances to similarity scores (1 - distance)
        # FAISS returns negative inner products for cosine similarity, so we negate
        similarities = -distances
        
        # Prepare results
        ids = []
        documents = []
        metadatas = []
        filtered_indices = []
        
        # Apply label filtering if needed
        if filter_labels:
            filter_matches = {label: 0 for label in filter_labels}
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_id = doc['id']
                    labels_found = False
                    
                    # Check document metadata for labels
                    if 'labels' in doc['metadata']:
                        doc_labels_str = doc['metadata']['labels']
                        try:
                            doc_labels = json.loads(doc_labels_str) if isinstance(doc_labels_str, str) else doc_labels_str
                            matches = [label for label in filter_labels if label in doc_labels]
                            if matches:
                                filtered_indices.append(i)
                                for match in matches:
                                    filter_matches[match] += 1
                                labels_found = True
                        except json.JSONDecodeError:
                            # If not valid JSON, treat as plain text
                            if any(label in doc_labels_str for label in filter_labels):
                                filtered_indices.append(i)
                                labels_found = True
                    
                    # Check enhanced metadata if available
                    if not labels_found and doc_id in self.metadata_store:
                        enhanced_metadata = self.metadata_store[doc_id]
                        if 'labels' in enhanced_metadata:
                            doc_labels = enhanced_metadata['labels']
                            matches = [label for label in filter_labels if label in doc_labels]
                            if matches:
                                filtered_indices.append(i)
                                for match in matches:
                                    filter_matches[match] += 1
                                    
                    # Limit to top-k results
                    if len(filtered_indices) >= k:
                        break
            
            # If no documents match filters, use unfiltered results
            if not filtered_indices:
                filtered_indices = list(range(min(k, len(indices[0]))))
        else:
            # No filtering, use all results up to k
            filtered_indices = list(range(min(k, len(indices[0]))))
        
        # Collect final results
        for i in filtered_indices:
            idx = indices[0][i]
            if idx < len(self.documents):
                doc = self.documents[idx]
                ids.append(doc['id'])
                documents.append(doc['text'])
                metadatas.append(doc['metadata'])
        
        return {
            'ids': [ids],
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [[similarities[0][i] for i in filtered_indices]]
        }
    
# ------------------------------
# Llama Model Configuration
# ------------------------------

class LlamaConfig:
    """Configuration for Meta Llama models with robust loading and fallback strategies."""
    # Model paths to Meta model directories
    LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/models/Llama3.2-3B")
    
    # Singleton instances
    _model = None
    _tokenizer = None
    
    # Configure model parameters
    GENERATION_MAX_TOKENS = 512
    GENERATION_TEMPERATURE = 0.7
    
    @classmethod
    def get_model(cls):
        """
        Load and return the Meta Llama model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model loading fails
        """
        if cls._model is None:
            logger = logging.getLogger(__name__)
            logger.info(f"Loading Meta Llama model from {cls.LLAMA_MODEL_PATH}")
            
            try:
                # Try approach 1: Use AutoModelForCausalLM (most flexible)
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    logger.info("Attempting to load model with AutoClasses")
                    cls._tokenizer = AutoTokenizer.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                    cls._model = AutoModelForCausalLM.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded model with AutoClasses")
                    return cls._model, cls._tokenizer
                except Exception as auto_err:
                    logger.warning(f"Failed to load with AutoClasses: {str(auto_err)}")
                    
                # Try approach 2: Use LlamaForCausalLM directly
                try:
                    from transformers import LlamaTokenizer, LlamaForCausalLM
                    
                    logger.info("Attempting to load model with LlamaForCausalLM directly")
                    cls._tokenizer = LlamaTokenizer.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                    cls._model = LlamaForCausalLM.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded model with LlamaForCausalLM")
                    return cls._model, cls._tokenizer
                except Exception as llama_err:
                    logger.warning(f"Failed to load with LlamaForCausalLM: {str(llama_err)}")
                
                # Try approach 3: Load with explicit path handling
                try:
                    import os
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    logger.info("Attempting to load model with explicit path handling")
                    # Check if the path exists
                    if not os.path.exists(cls.LLAMA_MODEL_PATH):
                        raise ValueError(f"Model path does not exist: {cls.LLAMA_MODEL_PATH}")
                    
                    # List files in the directory to help debug
                    files = os.listdir(cls.LLAMA_MODEL_PATH)
                    logger.info(f"Files in model directory: {files}")
                    
                    # Load with from_pretrained with explicit path handling
                    cls._tokenizer = AutoTokenizer.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        use_fast=False,  # Try disabling fast tokenizer
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                    # Load model with more conservative settings
                    cls._model = AutoModelForCausalLM.from_pretrained(
                        cls.LLAMA_MODEL_PATH,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded model with explicit path handling")
                    return cls._model, cls._tokenizer
                except Exception as path_err:
                    logger.warning(f"Failed to load with explicit path handling: {str(path_err)}")
                
                # All approaches failed
                raise RuntimeError("All model loading approaches failed")
                
            except Exception as e:
                error_msg = f"Error loading Meta Llama model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        return cls._model, cls._tokenizer
    
    @classmethod
    def get_embedding_dimension(cls):
        """Get the embedding dimension for the model."""
        try:
            model, _ = cls.get_model()
            return model.config.hidden_size
        except Exception:
            # Return default embedding dimension for Llama models
            return 4096

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
    description="Process all PDFs in batch to extract text, tables, and structured data",
    retries=1,                  # Retry once if the task fails
    retry_delay_seconds=60,     # Wait 60 seconds before retrying
    # Cache configuration could be added here if needed
)
def process_pdfs(pdf_metadata: Dict[str, Dict], 
                input_dir: str, 
                output_dir: str, 
                topic_definitions: Dict[str, str],
                reprocess_all: bool = False) -> Dict[str, Dict]:
    """
    Process all PDFs in the input directory by extracting text, tables, images, and structured data.
    Implements an incremental processing strategy that only processes new or changed files
    unless forced to reprocess everything.
    
    Args:
        pdf_metadata: Dictionary mapping filenames to their metadata including file paths and hashes
        input_dir: Directory containing the downloaded PDF files to process
        output_dir: Directory where processed files and extracted data will be saved
        topic_definitions: Dictionary of topic definitions used for content classification
        reprocess_all: Flag to force reprocessing of all PDFs even if unchanged
        
    Returns:
        Dictionary mapping filenames to their processing results including extracted content
    """
    # Get a Prefect logger for proper task logging within the Prefect UI
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist to avoid file write errors
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Path for the metadata file that tracks processing status across runs
    processed_path = os.path.join(output_dir, 'processed_metadata.json')
    
    # INCREMENTAL PROCESSING STRATEGY:
    # Load existing processed metadata if available to implement incremental processing
    # This allows us to skip unchanged files and only process new or modified PDFs
    if os.path.exists(processed_path) and not reprocess_all:
        logger.info(f"Loading existing processed metadata from {processed_path}")
        processed_metadata = pd.read_json(processed_path, orient='index').to_dict(orient='index')
        logger.info(f"Found {len(processed_metadata)} previously processed PDFs")
    else:
        logger.info("No existing metadata found or reprocessing all files")
        processed_metadata = {}
        
    # BUILD PROCESSING QUEUE:
    # Determine which files need processing based on their hash values
    # Files are selected for processing if they are:
    #  1. New (not in processed_metadata)
    #  2. Changed (hash differs from previously processed version)
    #  3. Force reprocessing is enabled (reprocess_all=True)
    to_process = []
    skipped = 0
    for filename, metadata in pdf_metadata.items():
        filepath = metadata['path']
        
        # Hash comparison to detect file changes
        if (filename in processed_metadata and 
            metadata['hash'] == processed_metadata[filename].get('hash') and
            not reprocess_all):
            # Skip unchanged files to save processing time
            skipped += 1
            continue
            
        # Queue file for processing with its metadata
        to_process.append((filename, filepath, metadata))
        
    logger.info(f"Processing {len(to_process)} PDFs (skipped {skipped} unchanged files)")
    
    # PARALLEL PROCESSING IMPLEMENTATION:
    # Process PDFs in parallel to maximize throughput
    # Uses ThreadPoolExecutor which is ideal for I/O-bound tasks like PDF processing
    with tqdm(total=len(to_process), desc="Processing PDFs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Create a mapping of futures to filenames for tracking results
            future_to_file = {
                # Submit each PDF for processing as a separate task
                executor.submit(process_pdf, filename, filepath, metadata, output_dir, topic_definitions): 
                filename for filename, filepath, metadata in to_process
            }
            
            # Collect results as they complete (in any order)
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    # Get the processing result for this PDF
                    result = future.result()
                    # Store the result in our metadata dictionary
                    processed_metadata[filename] = result
                    logger.debug(f"Successfully processed {filename}")
                except Exception as e:
                    # Log detailed error but continue processing other PDFs
                    logger.error(f"Error processing {filename}: {str(e)}")
                # Update progress bar after each PDF completes
                pbar.update(1)
                
    # PERSISTENCE FOR INCREMENTAL PROCESSING:
    # Save processed metadata to enable incremental processing in future runs
    # This creates a persistent record of which files have been processed and their hash values
    logger.info(f"Saving processed metadata to {processed_path}")
    pd.DataFrame.from_dict(processed_metadata, orient='index').to_json(
        processed_path, orient='index', indent=2
    )
    
    logger.info(f"Successfully processed {len(to_process)} PDFs")
    return processed_metadata

# ------------------------------
# Document Chunking Component
# ------------------------------

@task(
    name="chunk_document",                      # Task name in Prefect UI
    description="Chunk a document into semantic units for better retrieval",
    retries=2,                                  # Will retry twice if the task fails
    retry_delay_seconds=15                      # Wait 15 seconds between retries
    # No caching configuration as each document should be freshly processed
)
def chunk_document(filename: str, metadata: Dict, 
                  output_dir: str, 
                  topic_definitions: Dict[str, str]) -> List[Document]:
    """
    Chunk a single document into semantic units, label each chunk with relevant topics,
    and enrich chunks with metadata for improved retrieval. This function implements
    a two-stage chunking strategy that prioritizes preserving document structure.
    
    Args:
        filename: Name of the document file (used for identification and output paths)
        metadata: Complete metadata of the document including text content and structure
        output_dir: Directory where chunked results and labels will be saved
        topic_definitions: Dictionary mapping topic names to their descriptions for labeling
        
    Returns:
        List of chunked and labeled Document objects ready for vector storage
    """
    # Get Prefect logger for task-specific logging within Prefect UI
    logger = get_run_logger()
    
    # TOPIC CLASSIFICATION SETUP:
    # Initialize the topic classifier for semantic labeling of chunks
    # This creates vectorized representations of topic definitions for later comparison
    logger.debug(f"Initializing topic classifier with {len(topic_definitions)} topics")
    vectorizer, vectors, topic_names = initialize_topic_classifier(topic_definitions)
    
    # Extract essential document information from metadata
    text = metadata['text']                             # Full document text
    title = metadata['original_metadata']['title']      # Document title
    source_url = metadata['original_metadata']['url']   # Source URL
    
    # Get structured data if available for enriching chunk metadata
    # This allows us to associate extracted structured information with each chunk
    structured_data = metadata.get('structured_data', {})
    
    # CHUNKING CONFIGURATION:
    # Set parameters for text splitting to balance context and specificity
    chunk_size = 1000      # Target size of each chunk (characters)
    chunk_overlap = 200    # Overlap between chunks to maintain context across boundaries
    
    # DOCUMENT STRUCTURE-AWARE CHUNKING:
    # First try semantic chunking using predefined document sections
    # This preserves the logical structure of the document when possible
    chunks = []
    logger.debug(f"Attempting section-based chunking for {filename}")
    
    if 'sections' in metadata and metadata['sections']:
        section_count = 0
        for i, section in enumerate(metadata['sections']):
            section_text = section['text']
            section_title = section['title']
            
            # Skip very short sections which are likely just headings
            if len(section_text) < 100:
                logger.debug(f"Skipping short section: {section_title} ({len(section_text)} chars)")
                continue
                
            # Process each section into appropriately sized chunks with proper overlap
            # Each chunk retains knowledge of its source section for context
            section_chunks = split_text(
                text=section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata={
                    'source': filename,           # Original document filename
                    'title': title,               # Document title
                    'section': section_title,     # Section title
                    'section_idx': i,             # Section position in document
                    'url': source_url,            # Source URL
                    'chunk_type': 'section'       # Indicates this is a section-based chunk
                }
            )
            chunks.extend(section_chunks)
            section_count += 1
            
        logger.debug(f"Created {len(chunks)} chunks from {section_count} sections")
    
    # FALLBACK TO WHOLE DOCUMENT CHUNKING:
    # If document has no sections or too few chunks were created,
    # fall back to chunking the entire document as a whole
    if len(chunks) < 3:
        logger.info(f"Insufficient section-based chunks ({len(chunks)}), falling back to whole document chunking")
        chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={
                'source': filename,
                'title': title,
                'url': source_url,
                'chunk_type': 'document'        # Indicates this is a whole-document chunk
            }
        )
        logger.debug(f"Created {len(chunks)} chunks from whole document")
        
    # CHUNK ENHANCEMENT AND LABELING:
    # Enhance each chunk with additional metadata, classifications, and keywords
    # to improve search relevance and enable advanced filtering
    logger.debug(f"Enhancing and labeling {len(chunks)} chunks")
    labeled_chunks = []
    
    for i, chunk in enumerate(chunks):
        # TOPIC LABELING:
        # Label each chunk with relevant topics based on semantic similarity
        # This enables topic-based filtering during retrieval
        labels = label_chunk(chunk.page_content, vectorizer, vectors, topic_names)
        
        # Create a structured record of chunk labels for external use
        labeled_chunk = LabeledChunk(
            chunk_index=i,
            labels=labels
        )
        
        # METADATA ENRICHMENT:
        # Add indexing and relationship metadata
        chunk.metadata['chunk_idx'] = i                  # Position in chunk sequence
        chunk.metadata['total_chunks'] = len(chunks)     # Total chunks from this document
        chunk.metadata['labels'] = labels                # Topic labels for filtering
        
        # Add structured data references if available
        # This connects chunks to document-level structured information
        if structured_data:
            # Add core document metadata
            chunk.metadata['document_name'] = structured_data.get('document_name', '')
            chunk.metadata['document_date'] = structured_data.get('document_date', '')
            chunk.metadata['document_author'] = structured_data.get('document_author', '')
            
            # TOPIC ASSOCIATION:
            # Determine if this chunk is the primary source for any classified topics
            # This identifies "authoritative chunks" for each topic
            if 'topics' in metadata:
                for topic, topic_text in metadata['topics'].items():
                    # If the chunk contains the exact text snippet that defined a topic,
                    # mark this chunk as a primary source for that topic
                    if topic_text in chunk.page_content:
                        chunk.metadata['primary_topic'] = topic
                        logger.debug(f"Chunk {i} identified as primary for topic: {topic}")
                        break
        
        # KEYWORD EXTRACTION:
        # Extract representative keywords from the chunk
        # This enables keyword-based search and additional retrieval signals
        keywords = extract_keywords(chunk.page_content)
        chunk.metadata['keywords'] = keywords
        
        # Add to collection of processed chunks
        labeled_chunks.append(chunk)
        
        # PERSISTENCE:
        # Save labeled chunk data as JSON for external use and debugging
        # This makes the chunk metadata accessible outside the vector database
        labeled_chunk_path = os.path.join(
            output_dir, 
            f"{os.path.splitext(filename)[0]}_chunk_{i}_labels.json"
        )
        with open(labeled_chunk_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_chunk.dict(), f, indent=2)
        
    logger.info(f"Successfully processed {len(labeled_chunks)} chunks from {filename}")
    return labeled_chunks

@task(
    name="chunk_documents",                   # Prefect task name for UI display
    description="Chunk all documents in batch with incremental processing",
    retries=1,                                # Will retry once if the task fails
    retry_delay_seconds=30,                   # Wait 30 seconds before retry
    # Could add cache_key_fn and cache_expiration here if needed
)
def chunk_documents(processed_metadata: Dict[str, Dict], 
                   chunked_dir: str,
                   topic_definitions: Dict[str, str],
                   rechunk_all: bool = False) -> List[Document]:
    """
    Orchestrates the chunking process for all processed documents, implementing
    an incremental processing strategy that only processes new or changed documents.
    This function serves as the batch coordinator for the individual chunk_document tasks.
    
    Args:
        processed_metadata: Dictionary of document metadata from the processing stage
        chunked_dir: Output directory where chunked documents and metadata will be stored
        topic_definitions: Dictionary mapping topic names to descriptions for semantic labeling
        rechunk_all: Flag to force rechunking of all documents regardless of change status
        
    Returns:
        List of all document chunks created, ready for vector database indexing
    """
    # Get Prefect logger for task-specific logging
    logger = get_run_logger()
    
    # DIRECTORY SETUP:
    # Ensure output directory exists to prevent file write errors
    if not os.path.exists(chunked_dir):
        logger.info(f"Creating chunking output directory: {chunked_dir}")
        os.makedirs(chunked_dir)
    
    # PERSISTENCE CONFIGURATION:
    # Define path for metadata file that tracks chunking status across pipeline runs
    chunked_path = os.path.join(chunked_dir, 'chunked_metadata.json')
    
    # INCREMENTAL PROCESSING SETUP:
    # Load existing metadata if available to enable incremental processing
    # This allows us to skip rechunking documents that haven't changed
    if os.path.exists(chunked_path) and not rechunk_all:
        logger.info(f"Loading existing chunking metadata from {chunked_path}")
        chunked_metadata = pd.read_json(chunked_path, orient='index').to_dict(orient='index')
        logger.info(f"Found {len(chunked_metadata)} previously chunked documents")
    else:
        if rechunk_all:
            logger.info("Rechunking all documents (forced by rechunk_all flag)")
        else:
            logger.info("No existing chunking metadata found - processing all documents")
        chunked_metadata = {}
        
    # DOCUMENT SELECTION:
    # Identify documents that need chunking based on hash comparison
    # Documents are selected for chunking if they are:
    #  1. New (not in chunked_metadata)
    #  2. Modified (hash differs from previous chunking)
    #  3. Force rechunking is enabled (rechunk_all=True)
    to_chunk = []
    skipped_count = 0
    
    for filename, metadata in processed_metadata.items():
        # Check if document has been chunked before and is unchanged
        if (filename in chunked_metadata and 
            metadata['hash'] == chunked_metadata[filename].get('hash') and
            not rechunk_all):
            # Skip unchanged documents to save processing time
            skipped_count += 1
            continue
            
        # Queue document for chunking
        to_chunk.append((filename, metadata))
        
    logger.info(f"Chunking {len(to_chunk)} documents (skipped {skipped_count} unchanged documents)")
    
    # BATCH PROCESSING:
    # Process each document sequentially with progress tracking
    # Could be parallelized with concurrent.futures like in process_pdfs
    # but chunking is often more CPU-intensive than I/O-bound
    all_chunks = []  # Collects all chunks from all documents
    
    with tqdm(total=len(to_chunk), desc="Chunking documents") as pbar:
        for filename, metadata in to_chunk:
            try:
                # INDIVIDUAL DOCUMENT CHUNKING:
                # Process each document by calling the chunk_document task
                # This delegates the actual chunking work to a separate Prefect task
                logger.debug(f"Chunking document: {filename}")
                chunks = chunk_document(filename, metadata, chunked_dir, topic_definitions)
                
                # Accumulate all chunks for vector database indexing
                all_chunks.extend(chunks)
                
                # METADATA TRACKING:
                # Update chunking metadata to record successful processing
                # This enables incremental processing in future pipeline runs
                chunked_metadata[filename] = {
                    'hash': metadata['hash'],                    # Document hash for change detection
                    'chunking_date': datetime.now().isoformat(), # Timestamp for auditing/debugging
                    'num_chunks': len(chunks)                    # Number of chunks created
                }
                
                logger.debug(f"Created {len(chunks)} chunks from {filename}")
            except Exception as e:
                # ERROR HANDLING:
                # Log detailed error but continue processing other documents
                # This ensures partial progress even if some documents fail
                logger.error(f"Error chunking {filename}: {str(e)}")
                
            # Update progress bar after each document completes
            pbar.update(1)
            
    # PERSISTENCE FOR INCREMENTAL PROCESSING:
    # Save chunking metadata to enable incremental processing in future runs
    # This creates a persistent record of which documents have been chunked and their hash values
    logger.info(f"Saving chunking metadata to {chunked_path}")
    pd.DataFrame.from_dict(chunked_metadata, orient='index').to_json(
        chunked_path, orient='index', indent=2
    )
    
    # RESULTS SUMMARY:
    # Log overall statistics about the chunking process
    avg_chunks = len(all_chunks) / len(to_chunk) if to_chunk else 0
    logger.info(f"Created {len(all_chunks)} total chunks from {len(to_chunk)} documents (avg: {avg_chunks:.1f} chunks/doc)")
    
    return all_chunks

# ------------------------------
# Vector Database Component
# ------------------------------

@task(
    name="initialize_vector_db",
    description="Initialize or load the FAISS vector database",
    retries=2,
    retry_delay_seconds=15
)
def initialize_vector_db(db_dir: str, dimension: int = 4096) -> FAISSVectorStore:
    """
    Initialize or load the FAISS vector database.
    
    Args:
        db_dir: Directory to store vector database
        dimension: Dimension of the embedding vectors
        
    Returns:
        Initialized FAISS vector store
    """
    logger = get_run_logger()
    
    # Create directory if it doesn't exist
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    try:
        # Initialize FAISS vector store
        vector_store = FAISSVectorStore(db_dir, dimension)
        logger.info(f"Initialized FAISS vector database at {db_dir}")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing FAISS vector database: {str(e)}")
        raise

def embedding_cache_key_fn(context, parameters):
    """
    Generate a cache key for the embedding task based on:
    1. Document content hashes
    2. Database configuration
    3. Meta Llama model version and configuration
    
    Returns:
        str: A unique hash representing the task inputs
    """
    # Extract parameters
    documents = parameters["documents"]
    batch_size = parameters["batch_size"]
    
    # Generate a hash of document contents
    doc_hashes = []
    for doc in documents:
        # Hash the document content and key metadata
        doc_hash = sha256()
        doc_hash.update(doc.page_content.encode())
        
        # Add critical metadata to the hash
        if doc.metadata:
            # Only include metadata that affects embedding/retrieval
            critical_metadata = {
                k: v for k, v in doc.metadata.items() 
                if k in ['chunk_idx', 'source', 'labels']
            }
            doc_hash.update(json.dumps(critical_metadata, sort_keys=True).encode())
            
        doc_hashes.append(doc_hash.hexdigest())
    
    # Sort to ensure consistent order
    doc_hashes.sort()
    
    # Combine document hashes with other parameters
    combined_hash = sha256()
    combined_hash.update("_".join(doc_hashes).encode())
    combined_hash.update(str(batch_size).encode())
    
    # Add Meta Llama model information to detect model changes
    # Use the model path and timestamp for change detection
    model_path = os.environ.get("META_LLAMA_MODEL_PATH", "")
    
    # Get the modification time of the model directory or a key file in it
    # This helps detect when the model is updated
    model_timestamp = ""
    try:
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                # Check for key model files
                for key_file in ["params.json", "consolidated.00.pth", "tokenizer.model"]:
                    file_path = os.path.join(model_path, key_file)
                    if os.path.exists(file_path):
                        model_timestamp = str(os.path.getmtime(file_path))
                        break
            else:
                # If it's a file path, use that file's timestamp
                model_timestamp = str(os.path.getmtime(model_path))
    except:
        # If any error occurs, use empty string
        pass
    
    # Create a model version identifier that includes path and timestamp
    embedding_model_version = f"meta-llama-{os.path.basename(model_path)}-{model_timestamp}"
    combined_hash.update(embedding_model_version.encode())
    
    # Add PyTorch/CUDA version information as these can affect embeddings
    try:
        torch_version = f"{torch.__version__}-cuda{torch.version.cuda if torch.cuda.is_available() else 'none'}"
        combined_hash.update(torch_version.encode())
    except:
        pass
    
    return combined_hash.hexdigest()

def query_cache_key_fn(context, parameters):
    """
    Generate a cache key for the query processing task based on:
    1. Query text
    2. Database configuration
    3. Meta Llama model version and configuration
    
    Returns:
        str: A unique hash representing the task inputs
    """
    # Extract parameters
    query = parameters["query"]
    k = parameters.get("k", 5)
    use_structured_data = parameters.get("use_structured_data", True)
    
    # Generate a hash of the query
    query_hash = sha256()
    query_hash.update(query.encode())
    
    # Add other parameters that affect results
    query_hash.update(str(k).encode())
    query_hash.update(str(use_structured_data).encode())
    
    # Add Meta Llama model information to detect model changes
    # Use the model path and timestamp for change detection
    model_path = os.environ.get("META_LLAMA_MODEL_PATH", "")
    
    # Get the modification time of the model directory or a key file in it
    # This helps detect when the model is updated
    model_timestamp = ""
    try:
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                # Check for key model files
                for key_file in ["params.json", "consolidated.00.pth", "tokenizer.model"]:
                    file_path = os.path.join(model_path, key_file)
                    if os.path.exists(file_path):
                        model_timestamp = str(os.path.getmtime(file_path))
                        break
            else:
                # If it's a file path, use that file's timestamp
                model_timestamp = str(os.path.getmtime(model_path))
    except:
        # If any error occurs, use empty string
        pass
    
    # Create a model version identifier that includes path and timestamp
    embedding_model_version = f"meta-llama-{os.path.basename(model_path)}-{model_timestamp}"
    query_hash.update(embedding_model_version.encode())
    
    # Add PyTorch/CUDA version information as these can affect embeddings
    try:
        torch_version = f"{torch.__version__}-cuda{torch.version.cuda if torch.cuda.is_available() else 'none'}"
        query_hash.update(torch_version.encode())
    except:
        pass
    
    return query_hash.hexdigest()

@task(
    name="get_embeddings",                        # Descriptive name in Prefect UI
    description="Generate embeddings for text using Meta Llama model",
    retries=3,                                    # Retry 3 times on failure - helpful for occasional model errors
    retry_delay_seconds=5                         # Short delay between retries for transient errors
)
def get_embeddings(texts: List[str], model: str = "meta-llama") -> List[List[float]]:
    """
    Generate vector embeddings for a list of texts using Meta's Llama model.
    These embeddings are dense vector representations that capture semantic meaning,
    enabling similarity search and semantic matching in the vector database.
    
    Args:
        texts: List of text strings to convert to embeddings
        model: Model identifier (not used directly, just for logging)
        
    Returns:
        List of embedding vectors, each being a list of floating-point values
    """

    # LOGGING SETUP:
    # Get Prefect logger for structured task logging in the Prefect UI
    logger = get_run_logger()
    
    # INPUT VALIDATION:
    # Check for empty input to avoid unnecessary model calls and potential errors
    if not texts:
        logger.warning("Received empty text list for embedding generation - returning empty list")
        return []
        
    try:
        # EMBEDDING GENERATION:
        # Get the Meta Llama model and tokenizer from our configuration
        model, tokenizer = LlamaConfig.get_model()
        
        # Log what we're about to do
        logger.debug(f"Generating embeddings for {len(texts)} texts using Meta Llama model")
        
        # MODEL CALL:
        # Process text with Meta Llama model to create embeddings
        # We'll use the mean of the last hidden layer as our embedding vector
        embeddings = []
        
        for text in texts:
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get the model's hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            # Extract the last hidden state (the output of the last transformer layer)
            # Shape is typically [batch_size, sequence_length, hidden_size]
            last_hidden_state = outputs.hidden_states[-1]
            
            # Mean pooling - take the average over token dimension
            # This gives us one vector per input text
            mean_embedding = last_hidden_state.mean(dim=1)
            
            # Convert to regular Python list and add to results
            embedding = mean_embedding[0].cpu().numpy().tolist()
            embeddings.append(embedding)
        
        # Verify we got the expected number of embeddings back
        if len(embeddings) != len(texts):
            logger.warning(f"Expected {len(texts)} embeddings but received {len(embeddings)}")
            
        logger.debug(f"Successfully generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
        
        # QUALITY CHECK:
        # Verify embeddings are non-zero (a zero vector would indicate possible failure)
        if embeddings and all(sum(abs(x) for x in emb) < 0.01 for emb in embeddings):
            logger.warning("Generated embeddings have very low magnitude - possible quality issue")
        
        return embeddings
        
    except RuntimeError as e:
        # MODEL LOADING ERROR:
        # Handle issues with loading or running the Llama model
        logger.error(f"Meta Llama model runtime error: {str(e)}")
        logger.info("Check if the model path is correct and the model files exist")
        # Re-raise to trigger Prefect retry mechanism
        raise
        
    except torch.cuda.OutOfMemoryError as e:
        # GPU MEMORY ERROR:
        # Specific handling for GPU out of memory errors
        logger.error(f"GPU memory error while generating embeddings: {str(e)}")
        logger.critical("Consider using a smaller model, reducing batch size, or using CPU instead")
        raise  # Memory errors likely won't resolve with retries
        
    except MemoryError as e:
        # MEMORY ERROR:
        # Meta Llama models can be memory-intensive
        logger.error(f"Memory error while generating embeddings: {str(e)}")
        logger.critical("Consider using a smaller model or reducing batch size")
        raise  # Memory errors likely won't resolve with retries
        
    except Exception as e:
        # GENERAL ERROR HANDLER:
        # Catch any other unexpected errors for robustness
        logger.error(f"Unexpected error generating embeddings: {type(e).__name__}: {str(e)}")
        
        # FALLBACK STRATEGY:
        # Return zero embeddings as a last resort to prevent pipeline failure
        # In production systems, this maintains pipeline integrity at cost of search quality
        # Zero vectors won't match anything in semantic search
        logger.warning(f"Returning zero embeddings due to model error - search quality will be affected")
        
        # Get the embedding dimension from the model configuration
        try:
            embedding_dim = model.config.hidden_size
        except:
            embedding_dim = 4096  # Default embedding dimension for Llama models
            
        # Generate zero vectors with correct dimensionality
        # Using zeros maintains the expected data structure for downstream processing
        return [[0.0] * embedding_dim for _ in texts]
    
@task(
    name="get_embeddings_batch",                     # Task name for Prefect UI monitoring
    description="Generate embeddings for texts in batches with memory management for Meta Llama models",
    retries=2,                                       # Retry twice on failure - balances resilience with progress
    retry_delay_seconds=10,                          # Longer delay between retries to allow model to reset
    # No caching by default as embeddings should be fresh; add cache_key_fn for dev/test environments if needed
)
def get_embeddings_batch(texts: List[str], 
                         batch_size: int = 4,       # Even more conservative default for Meta models 
                         model: str = "meta-llama") -> List[List[float]]:
    """
    Process a large collection of texts into embeddings by breaking them into smaller batches
    to manage memory usage and error resilience. This is essential for processing document 
    collections of significant size when using Meta Llama models, which can be memory-intensive.
    
    Args:
        texts: Complete list of text chunks to convert to embeddings
        batch_size: Number of texts to process in each batch (smaller for Meta Llama due to memory constraints)
        model: Model identifier (just for logging, actual model loaded from LlamaConfig)
        
    Returns:
        List of embedding vectors for all input texts, preserving original order
    """

    # LOGGING INITIALIZATION:
    # Get Prefect logger for structured task tracking in Prefect UI
    logger = get_run_logger()
    logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    # BATCH PROCESSING SETUP:
    # Initialize container for all embeddings, preserving input text order
    # Order preservation is critical for mapping embeddings back to documents
    all_embeddings = []
    
    # Pre-calculate batch statistics for logging and progress tracking
    total_batches = (len(texts) + batch_size - 1) // batch_size
    estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough token estimate
    logger.info(f"Processing approximately {estimated_tokens:.0f} tokens across {total_batches} batches")
    
    # MEMORY MANAGEMENT PLANNING:
    # Calculate batch pause based on expected computational load
    # Meta Llama models can be very memory-intensive so we need proper memory management
    if total_batches > 5:
        logger.info(f"Large job detected ({total_batches} batches) - implementing aggressive memory management")
        batch_pause = 2  # Longer pause for large jobs with Meta models
    else:
        batch_pause = 1  # Standard pause for smaller jobs
    
    # Get embedding dimension for zero vectors if needed
    embedding_dim = None
    try:
        # Try to get the model to determine embedding dimension
        model_obj, _ = LlamaConfig.get_model()
        embedding_dim = model_obj.config.hidden_size
    except Exception:
        # Default to typical Llama embedding dimension if we can't get it from model
        embedding_dim = 4096
        logger.debug(f"Using default embedding dimension of {embedding_dim}")
    
    # BATCH PROCESSING LOOP:
    # Process in batches with visual progress tracking via tqdm
    # This provides real-time feedback during long-running embedding operations
    with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            # Extract the current batch while maintaining original indices
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # BATCH CONTEXT LOGGING:
            # Log detailed batch information for monitoring and debugging
            # This helps track progress and diagnose issues in specific batches
            logger.debug(f"Processing embedding batch {batch_num}/{total_batches} " +
                       f"(items {i} to {min(i+batch_size-1, len(texts)-1)})")
            
            # BATCH LENGTH ANALYSIS:
            # Log batch text length statistics to help diagnose potential issues
            # Extremely long texts might cause context length issues with the model
            if logger.isEnabledFor(logging.DEBUG):
                lengths = [len(text) for text in batch]
                logger.debug(f"Batch text lengths - min: {min(lengths)}, " +
                           f"max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.1f} chars")
            
            try:
                # EMBEDDING GENERATION:
                # Call the individual embedding function for this batch
                # This delegates to the more focused get_embeddings function
                batch_embeddings = get_embeddings(batch, model)
                
                # VALIDATION:
                # Verify batch results match expectations before proceeding
                if len(batch_embeddings) != len(batch):
                    logger.warning(f"Batch size mismatch: expected {len(batch)} embeddings, " +
                                 f"got {len(batch_embeddings)} - data inconsistency may occur")
                
                # Accumulate results in order
                all_embeddings.extend(batch_embeddings)
                
                # MEMORY MANAGEMENT:
                # More aggressive memory management for Meta models
                if i + batch_size < len(texts):
                    # Only pause if we have more batches to process
                    logger.debug(f"Pausing {batch_pause}s for memory management")
                    time.sleep(batch_pause)
                    
                    # Force garbage collection after each batch for Meta models
                    import gc
                    
                    # Clear CUDA cache if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # Run garbage collection
                    gc.collect()
                    
            except torch.cuda.OutOfMemoryError as e:
                # SPECIFIC GPU MEMORY ERROR HANDLING:
                # Special case for GPU out of memory errors
                logger.error(f"GPU out of memory in batch {batch_num}/{total_batches}: {str(e)}")
                logger.info("Attempting to recover by clearing CUDA cache and reducing batch size")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                gc.collect()
                
                # DATA CONSISTENCY STRATEGY:
                # Return zero embeddings for this batch to maintain index alignment
                logger.warning(f"Using zero vectors for batch {batch_num} due to GPU memory error")
                zero_embeddings = [[0.0] * embedding_dim for _ in batch]
                all_embeddings.extend(zero_embeddings)
                
                # ADAPTIVE BATCH MANAGEMENT:
                # Aggressively reduce batch size after GPU OOM errors
                old_batch_size = batch_size
                batch_size = max(1, batch_size // 2)  # More aggressive reduction, allow batch size of 1
                logger.debug(f"Reducing batch size from {old_batch_size} to {batch_size} after GPU OOM")
                
                # Increase pause significantly after GPU errors
                batch_pause = min(batch_pause * 3, 10)  # Exponential backoff up to 10 seconds
                logger.debug(f"Increasing batch pause to {batch_pause}s after GPU OOM")
                
            except Exception as e:
                # GENERAL ERROR HANDLING:
                # Handle batch failures while allowing the pipeline to continue
                # This prevents a single batch error from failing the entire job
                logger.error(f"Error in embedding batch {batch_num}/{total_batches}: {type(e).__name__}: {str(e)}")
                
                # MEMORY RECLAMATION ON ERROR:
                # Force garbage collection to reclaim memory after errors
                import gc
                gc.collect()
                
                # Try to clear CUDA cache if available
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                
                # DATA CONSISTENCY STRATEGY:
                # Return zero embeddings for this batch to maintain index alignment
                # This ensures downstream processing can continue with degraded data
                logger.warning(f"Using zero vectors for batch {batch_num} due to error - " +
                             "search quality for these items will be affected")
                zero_embeddings = [[0.0] * embedding_dim for _ in batch]
                all_embeddings.extend(zero_embeddings)
                
                # ADAPTIVE BATCH MANAGEMENT:
                # Reduce batch size after errors to avoid repeated failures
                if batch_size > 2:
                    old_batch_size = batch_size
                    batch_size = max(2, batch_size // 2)  # Reduce batch size but keep at least 2
                    logger.debug(f"Reducing batch size from {old_batch_size} to {batch_size} after error")
                
                # INCREASE PAUSE AFTER ERRORS:
                # Allow more time for model to reset after errors
                batch_pause = min(batch_pause * 2, 5)  # Exponential backoff up to 5 seconds
                logger.debug(f"Increasing batch pause to {batch_pause}s after error")
                
            # Update progress bar after each batch completes
            pbar.update(len(batch))
    
    # VALIDATION AND SUMMARY:
    # Verify overall job results match expectations
    if len(all_embeddings) != len(texts):
        logger.error(f"Total embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}")
    
    # Calculate zero vectors to assess potential quality issues
    zero_count = sum(1 for emb in all_embeddings if sum(abs(x) for x in emb) < 0.01)
    if zero_count > 0:
        logger.warning(f"{zero_count} zero/near-zero embeddings detected out of {len(all_embeddings)} " +
                     f"({zero_count/len(all_embeddings)*100:.1f}%) - search quality will be affected")
    
    logger.info(f"Successfully completed embedding generation for {len(texts)} texts")
    return all_embeddings

@task(
    name="embed_and_index_documents",
    description="Create embeddings and index documents in FAISS vector database",
    cache_key_fn=embedding_cache_key_fn,
    cache_expiration=timedelta(hours=24),
    retries=2,
    retry_delay_seconds=30
)
def embed_and_index_documents(documents: List[Document], 
                             vector_store: FAISSVectorStore, 
                             db_dir: str,
                             batch_size: int = 32) -> None:
    """
    Generate embeddings for chunked documents and store them in the FAISS vector database
    along with their metadata.
    
    Args:
        documents: List of chunked Document objects ready for embedding and indexing
        vector_store: Initialized FAISS vector store
        db_dir: Directory where vector database and metadata are stored
        batch_size: Number of documents to process in each batch
    """
    # Get Prefect logger for structured task logging
    logger = get_run_logger()
    logger.info(f"Indexing {len(documents)} documents in FAISS vector database")
    
    # BATCH PROCESSING SETUP:
    # Calculate number of batches for progress tracking and rate limiting
    total_batches = (len(documents) + batch_size - 1) // batch_size
    logger.info(f"Processing in {total_batches} batches of up to {batch_size} documents each")
    
    # DOCUMENT INDEXING LOOP:
    # Process documents in batches with progress tracking
    with tqdm(total=len(documents), desc="Indexing documents") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # DOCUMENT PREPARATION:
                # Prepare document data for indexing in vector database
                # Generate unique IDs for each document in the batch
                ids = [f"doc_{i+j}" for j in range(len(batch))]
                
                # Extract text content from documents for embedding generation
                texts = [doc.page_content for doc in batch]
                
                # METADATA HANDLING:
                metadatas = []
                
                for j, doc in enumerate(batch):
                    doc_id = ids[j]
                    
                    # Process metadata
                    metadata = {}
                    
                    # Process each metadata field according to its type
                    for key, value in doc.metadata.items():
                        # For lists, convert to string representation
                        if isinstance(value, list):
                            # Convert list to string representation for storage
                            metadata[key] = json.dumps(value)
                        # SIMPLE TYPES: Store directly
                        elif isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        # COMPLEX TYPES: Store as JSON string
                        elif isinstance(value, dict):
                            metadata[key] = json.dumps(value)
                        # For any other types, convert to string
                        else:
                            try:
                                metadata[key] = str(value)
                            except:
                                # If conversion fails, skip this metadata
                                logger.warning(f"Skipping metadata key {key} with unsupported type {type(value)}")
                    
                    # Add metadata to batch
                    metadatas.append(metadata)
                
                # EMBEDDING GENERATION:
                # Convert text content to vector embeddings
                logger.debug(f"Generating embeddings for {len(texts)} documents")
                embeddings = get_embeddings_batch(texts, batch_size=min(batch_size, 8))
                
                # DATABASE INSERTION:
                # Add documents, embeddings, and metadata to FAISS
                vector_store.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.debug(f"Added batch {batch_num} to FAISS vector database")
                
                # MEMORY MANAGEMENT:
                # Pause between batches to allow memory to be reclaimed
                if i + batch_size < len(documents):
                    logger.debug("Pausing to allow memory reclamation")
                    time.sleep(1)
                    
                    # Force garbage collection to ensure memory is freed
                    import gc
                    gc.collect()
                
            except Exception as e:
                # ERROR HANDLING:
                # Log detailed error but continue with next batch
                logger.error(f"Error indexing batch {batch_num}/{total_batches}: {str(e)}")
                logger.debug(f"Batch size: {len(batch)}, First ID: {ids[0] if ids else 'N/A'}")
                
                # MEMORY CLEANUP AFTER ERROR:
                # Ensure memory is freed after errors
                import gc
                gc.collect()
                
            # Update progress bar after each batch completes
            pbar.update(len(batch))
    
    # COMPLETION SUMMARY:
    # Log final indexing statistics
    logger.info(f"Successfully indexed {len(documents)} documents in the FAISS vector database")

def search_cache_key_fn(context, parameters):
    """
    Generate a cache key for the semantic search function that excludes
    non-serializable objects like the ChromaDB client.
    """
    # Extract serializable parameters
    query = parameters["query"]
    db_dir = parameters["db_dir"]
    k = parameters.get("k", 5)
    filter_labels = parameters.get("filter_labels", None)
    
    # Create a hash of the serializable parameters
    key_hash = sha256()
    key_hash.update(query.encode())
    key_hash.update(db_dir.encode())
    key_hash.update(str(k).encode())
    
    # Add filter labels if present
    if filter_labels:
        key_hash.update(json.dumps(sorted(filter_labels)).encode())
    
    # Return the hexadecimal digest
    return key_hash.hexdigest()
    
@task(
    name="semantic_search",
    description="Perform semantic search in the FAISS vector database"
)
def semantic_search(query: str, 
                   vector_store: FAISSVectorStore, 
                   db_dir: str,
                   k: int = 5,
                   filter_labels: List[str] = None) -> List[Document]:
    """
    Execute a semantic similarity search against the FAISS vector database to find
    contextually relevant documents based on meaning rather than keyword matching.
    
    Args:
        query: User's query text to find semantically similar documents for
        vector_store: Initialized FAISS vector store
        db_dir: Directory where vector database and enhanced metadata are stored
        k: Maximum number of results to return
        filter_labels: Optional list of topic labels to filter results by
        
    Returns:
        List of Document objects containing the most relevant content and metadata
    """
    # Get Prefect logger for structured task logging
    logger = get_run_logger()
    
    # Log the count of documents in vector store
    doc_count = vector_store.count()
    logger.info(f"Vector store contains {doc_count} documents")
    
    if doc_count == 0:
        logger.warning("Vector database is empty. No documents to search.")
        return []
    
    # QUERY EMBEDDING GENERATION:
    # Convert the textual query into the same vector space as our documents
    logger.debug(f"Generating embedding for query: '{query}'")
    query_embedding = get_embeddings([query])[0]
    
    # SEMANTIC SEARCH EXECUTION:
    try:
        # Perform search in FAISS
        logger.info(f"Executing semantic search for query: '{query}'")
        if filter_labels:
            logger.info(f"Applying label filters: {filter_labels}")
            
        results = vector_store.search(
            embedding=query_embedding,
            k=k,
            filter_labels=filter_labels
        )
        
        # RESULT CONSTRUCTION:
        # Convert raw FAISS results into Document objects for downstream processing
        logger.debug(f"Converting {len(results['documents'][0])} results to Document objects")
        docs = []
        
        for i in range(len(results['documents'][0])):
            # Get document data
            doc_id = results['ids'][0][i] if 'ids' in results else f"result_{i}"
            doc_text = results['documents'][0][i]
            doc_metadata = results['metadatas'][0][i].copy() if results['metadatas'][0][i] else {}
            
            # METADATA PARSING:
            # Convert stored JSON strings back to original Python objects
            for key, value in list(doc_metadata.items()):
                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    try:
                        # Try to parse JSON strings back to Python objects
                        doc_metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # If parsing fails, keep as string
                        pass
            
            # METADATA ENRICHMENT:
            # Add enhanced metadata from vector store
            if hasattr(vector_store, 'metadata_store') and doc_id in vector_store.metadata_store:
                for key, value in vector_store.metadata_store[doc_id].items():
                    doc_metadata[key] = value
            
            # Add relevance score
            distance = results['distances'][0][i] if i < len(results['distances'][0]) else 0
            doc_metadata['relevance_score'] = distance
            
            # Create standardized Document object with content and metadata
            doc = Document(
                page_content=doc_text,
                metadata=doc_metadata
            )
            docs.append(doc)
        
        logger.info(f"Semantic search completed with {len(docs)} results for query: '{query}'")
        return docs
        
    except Exception as e:
        # ERROR HANDLING:
        logger.error(f"Error in semantic search: {type(e).__name__}: {str(e)}")
        logger.warning("Returning empty results due to search error")
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
    description="Generate answer from context documents using Meta Llama model"
)
def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    Generate an answer for the query using retrieved context documents and the Meta Llama model.
    
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
    
    # Prepare prompt for the Meta Llama model
    # Format for Meta Llama models (may differ slightly based on specific model version)
    prompt = f"""<s>[INST] <<SYS>>
You are a helpful mining industry expert that provides accurate answers based only on the context provided. 
If you don't find the answer in the context, say "I don't have enough information to answer this question."
Only use the information provided in the context below.
<</SYS>>

Context:
{context}

Question: {query} [/INST]"""
    
    try:
        # Get the Meta Llama model and tokenizer from our configuration
        model, tokenizer = LlamaConfig.get_model()
        
        # Call Meta Llama model to generate answer
        logger.debug(f"Generating answer with Meta Llama model for query: '{query}'")
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response with appropriate parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=LlamaConfig.GENERATION_MAX_TOKENS,
                temperature=LlamaConfig.GENERATION_TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract answer (everything after the prompt)
        answer = generated_text[len(prompt):].strip()
        
        # For some Meta Llama models, the output might contain additional tokens
        # Clean up by removing common end-of-answer markers
        answer_markers = ["</s>", "[/INST]", "<|endoftext|>"]
        for marker in answer_markers:
            if marker in answer:
                answer = answer.split(marker)[0].strip()
        
        return answer
    except torch.cuda.OutOfMemoryError as e:
        # Specific handling for GPU memory errors
        logger.error(f"GPU memory error while generating answer: {str(e)}")
        
        # Clear GPU cache and run garbage collection
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return "I encountered a resource limitation while generating an answer. Please try simplifying your query or reducing the amount of context."
    except Exception as e:
        logger.error(f"Error generating answer with Meta Llama: {str(e)}")
        
        # Memory reclamation in case of GPU/memory issues
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
        import gc
        gc.collect()
        
        return f"An error occurred while generating the answer: {str(e)}"
    
# Update the process_query function to use our new FAISS vector store
@task(
    name="process_query",
    description="Process a user query and generate an answer",
    cache_key_fn=query_cache_key_fn,
    cache_expiration=timedelta(hours=24)
)
def process_query(query: str, 
                 vector_store: FAISSVectorStore,
                 db_dir: str,
                 topic_definitions: Dict[str, str],
                 k: int = 5,
                 use_structured_data: bool = True) -> Dict[str, Any]:
    """
    Process a query using the RAG system with structured data.
    
    Args:
        query: Query text
        vector_store: FAISS vector store
        db_dir: Directory where vector database is stored
        topic_definitions: Dictionary of topic definitions for classification
        k: Number of context documents to retrieve
        use_structured_data: Whether to use structured data in response
        
    Returns:
        Dictionary with query results, sources, and structured data
    """
    logger = get_run_logger()
    logger.info(f"Processing query: '{query}'")

    # Check if documents exist in the vector store
    count = vector_store.count()
    logger.info(f"Documents in vector database: {count}")

    # Check if vector store is empty
    if count == 0:
        return {
            'query': query,
            'answer': "There are no documents in the vector database to search.",
            'sources': [],
            'topics': []
        }

    # Step 1: Classify query into topics for targeted retrieval
    query_topics = classify_query(query, topic_definitions)
    logger.info(f"Query topics: {query_topics}")

    # Testing embedding generation
    logger.info("Testing embedding generation...")
    test_embeddings = get_embeddings([query])
    logger.info(f"Test embedding dimension: {len(test_embeddings[0])}")

    # Step 2: Get context documents, filtered by topic if available
    if query_topics and use_structured_data:
        context_docs = semantic_search(
            query, vector_store, db_dir, k=k, filter_labels=query_topics
        )
    else:
        logger.info("Searching without topic filtering")
        context_docs = semantic_search(query, vector_store, db_dir, k=k)
    
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
    name="msi_pipeline",
    description="Full RAG pipeline from scraping to indexing",
    log_prints=True
)
def msi_pipeline(base_url: str, config: Dict = None):
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
    name="update_msi_pipeline",
    description="Update RAG pipeline with new or changed documents",
    log_prints=True
)
def update_msi_pipeline(base_url: str, config: Dict = None):
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
    name="query_msi_system",
    description="Query the RAG system and get an answer",
    log_prints=True
)
def query_msi_system(query: str, db_dir: str = "vector_db", k: int = 5):
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
        vector_store=client,  # Using the initialized client variable
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

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Setup Meta Llama model path
    # This should point to the directory containing the Meta Llama model files
    # (checkpoint.chk, consolidated.00.pth, params.json, tokenizer.model)
    if not os.environ.get("META_LLAMA_MODEL_PATH"):
        os.environ["META_LLAMA_MODEL_PATH"] = "/models/Llama3.2-3B"
    
    # Set device configuration - uncomment the appropriate line below
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use specific GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Force CPU only mode
    
    # Set up memory efficient loading for very large models
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Optional: Prevent tokenizers from using multiple threads which can cause issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Configuration for the RAG pipeline
    config = {
        'download_dir': '1_downloaded_pdfs',
        'processed_dir': '2_processed_pdfs',
        'chunked_dir': '3_chunked_docs',
        'vector_db_dir': '4_vector_db',
        'max_pdfs': 20,
        'max_pages': 5,
        'reprocess': False,
        # Added Meta Llama specific configurations
        'model_max_length': 2048,     # Maximum tokens per document chunk
        'embedding_batch_size': 4,     # Smaller batch size for Meta models
        'use_4bit_quantization': True  # Enable 4-bit quantization to reduce memory usage
    }
    
    # Import and initialize PyTorch/transformers before running the pipeline
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Print information about available CUDA devices
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available, using CPU. This may be very slow for large models.")
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Please install with: pip install torch transformers")
        exit(1)
    
    print(f"Using Meta Llama model at: {os.environ.get('META_LLAMA_MODEL_PATH')}")
    
    # Run the pipeline
    try:
        result = msi_pipeline("https://evolutionmining.com.au", config)
        
        # Query
        answer = query_msi_system(
            "What was total gold production in tons in 2024?",
            db_dir=result['db_dir']
        )
        
        print("\nDone!")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        
        # Print more detailed error information
        import traceback
        traceback.print_exc()