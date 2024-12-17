from pydantic import BaseModel
from pydantic import BaseModel
from typing import List


class MetaData(BaseModel):
    category: str


class TableDescription(BaseModel):
    description: str


class ChunkGroupSchema(BaseModel):
    """
    JSON schema for a single chunk group.
    """
    reason: str
    chunk_id: int
    sentences: List[int]


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
