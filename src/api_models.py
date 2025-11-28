"""API models for the RAG server."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    
    question: str = Field(
        ...,
        min_length=1,
        description="The question to answer",
        examples=["What is the capital of France?"]
    )


class Source(BaseModel):
    """Model for a source reference."""
    
    number: int = Field(
        ...,
        description="Source reference number",
        ge=1
    )
    title: str = Field(
        ...,
        description="Title of the source article"
    )
    url: str = Field(
        ...,
        description="URL of the source article"
    )


class AnswerResponse(BaseModel):
    """Response model for an answer."""
    
    answer: str = Field(
        ...,
        description="The generated answer to the question"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the question in seconds",
        ge=0
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ...,
        description="Server status",
        examples=["healthy"]
    )
    ollama_accessible: bool = Field(
        ...,
        description="Whether Ollama service is accessible"
    )
    message: Optional[str] = Field(
        None,
        description="Additional status message"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

