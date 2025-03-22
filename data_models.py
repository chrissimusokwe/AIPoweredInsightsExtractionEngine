from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Union
from camelot.core import TableList

# Pydantic Model for Structured Extraction
class ExtractedData(BaseModel):
    document_name: str
    document_date: Optional[str] = None
    document_author: Optional[str] = None
    topics: Dict[str, str]

# Pydantic Model for Labeled Chunk Output
class LabeledChunk(BaseModel):
    chunk_index: int
    labels: List[str]

# Define Pydantic models for structured output
class TableSchema(BaseModel):
    title: str
    data: List[Dict[str, Union[str, float, int]]] = Field(default_factory=list)

class ImageSchema(BaseModel):
    url: str
    description: str
    mine_site: str
    category: str  # e.g., "processing plant", "open pit", etc.

class ParsedDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    tables: List[TableList]
    hyperlinks: List[str]
    images: List[ImageSchema]

# Pydantic Model for JSON Resources and Reserves Data
class Grade(BaseModel):
    """Grade values for different metals (e.g., Au, Cu, Zn)"""
    values: Dict[str, float]  # Dictionary where keys are metal names and values are grades

class Category(BaseModel):
    """Represents ore tonnage, contained metal, and grade for a resource or reserve category."""
    ore: Optional[float] = None  # Ore tonnage in metric tonnes
    metal: Optional[float] = None  # Contained metal in metric tonnes
    grade: Optional[Grade] = None  # Grade per metal

class Method(BaseModel):
    """Mining method (e.g., Open Pit, Underground, Stockpile). Not all mine sites are split per mining method."""
    method: Optional[str] = None  # Mining method name
    measured: Optional[Category] = None
    indicated: Optional[Category] = None
    inferred: Optional[Category] = None
    proven: Optional[Category] = None
    probable: Optional[Category] = None

class Deposit(BaseModel):
    """Deposit within a mine site, normally proper names or codes. Not all mine sites are split per deposit."""
    deposit: Optional[str] = None  # Name of the deposit
    methods: List[Method] = Field(default_factory=list)  # Use default_factory for lists

class MineSite(BaseModel):
    """Mine site containing different deposits."""
    mine_site: str  # Name of the mine site
    deposits: List[Deposit] = Field(default_factory=list)  # Use default_factory for lists

class ReservesAndResources(BaseModel):
    """Top-level model to hold both resources and reserves."""
    resources: List[MineSite] = Field(default_factory=list)
    reserves: List[MineSite] = Field(default_factory=list)