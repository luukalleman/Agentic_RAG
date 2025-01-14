from pydantic import BaseModel
from pydantic import BaseModel
from typing import List


class MetaData(BaseModel):
    category: str


class TableDescription(BaseModel):
    description: str


from pydantic import BaseModel
from typing import List

class ChunkGroupSchema(BaseModel):
    """
    JSON schema for a single chunk group with actual sentence text.
    """
    reason: str
    chunk_id: int
    sentences: List[str]  # Changed to store actual text instead of IDs

class ChunkGroups(BaseModel):
    """
    JSON schema for the structured output of all chunk groups.
    """
    chunks: List[ChunkGroupSchema]


class ChunkGroupSchemaDirect(BaseModel):
    """
    JSON schema for a single chunk group.
    """
    sentences: str


class ChunkGroupsDirect(BaseModel):
    """
    JSON schema for the structured output of all chunk groups.
    """
    chunks: List[ChunkGroupSchemaDirect]
