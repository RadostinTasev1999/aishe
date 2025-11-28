"""API client for the RAG server."""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

import httpx


@dataclass
class APIAnswer:
    """Answer from the API."""
    answer: str
    sources: List[Dict[str, str]]
    processing_time: float


@dataclass
class APIHealth:
    """Health status from the API."""
    status: str
    ollama_accessible: bool
    message: Optional[str] = None


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class ServerNotReachableError(APIClientError):
    """Raised when the server is not reachable."""
    pass


class ServerError(APIClientError):
    """Raised when the server returns an error."""
    pass


class RAGAPIClient:
    """Client for the RAG API server."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 120.0
    ):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the API server. If None, uses AISHE_API_URL
                     environment variable or defaults to http://localhost:8000
            timeout: Request timeout in seconds (default: 120s for LLM processing)
        """
        if base_url is None:
            base_url = os.getenv("AISHE_API_URL", "http://localhost:8000")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def check_health(self) -> APIHealth:
        """Check the health of the API server.
        
        Returns:
            APIHealth object with server status
            
        Raises:
            ServerNotReachableError: If the server is not reachable
            ServerError: If the server returns an error
        """
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            return APIHealth(
                status=data["status"],
                ollama_accessible=data["ollama_accessible"],
                message=data.get("message")
            )
        except httpx.ConnectError as e:
            raise ServerNotReachableError(
                f"Cannot connect to server at {self.base_url}. "
                "Make sure the server is running."
            ) from e
        except httpx.TimeoutException as e:
            raise ServerNotReachableError(
                f"Request to {self.base_url} timed out."
            ) from e
        except httpx.HTTPStatusError as e:
            raise ServerError(
                f"Server returned error: {e.response.status_code}"
            ) from e
        except Exception as e:
            raise APIClientError(f"Unexpected error: {str(e)}") from e
    
    def ask_question(self, question: str) -> APIAnswer:
        """Ask a question to the RAG system.
        
        Args:
            question: The question to ask
            
        Returns:
            APIAnswer object with the answer and sources
            
        Raises:
            ServerNotReachableError: If the server is not reachable
            ServerError: If the server returns an error
            APIClientError: For other errors
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/v1/ask",
                json={"question": question.strip()}
            )
            response.raise_for_status()
            data = response.json()
            
            # Convert sources to dict format
            sources = [
                {
                    "number": source["number"],
                    "title": source["title"],
                    "url": source["url"]
                }
                for source in data["sources"]
            ]
            
            return APIAnswer(
                answer=data["answer"],
                sources=sources,
                processing_time=data["processing_time"]
            )
        except httpx.ConnectError as e:
            raise ServerNotReachableError(
                f"Cannot connect to server at {self.base_url}. "
                "Make sure the server is running with: nix run .#server"
            ) from e
        except httpx.TimeoutException as e:
            raise ServerNotReachableError(
                f"Request timed out after {self.timeout} seconds. "
                "The question might be too complex."
            ) from e
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(error_data))
            except:
                error_detail = e.response.text
            
            raise ServerError(
                f"Server error ({e.response.status_code}): {error_detail}"
            ) from e
        except ValueError as e:
            raise e
        except Exception as e:
            raise APIClientError(f"Unexpected error: {str(e)}") from e

