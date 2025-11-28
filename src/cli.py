"""Command-line interface for the RAG system."""

import sys
from pathlib import Path

from api_client import RAGAPIClient, ServerNotReachableError, ServerError, APIClientError


class RAGCLI:
    """Command-line interface for RAG question answering."""

    def __init__(self, api_url=None):
        """Initialize the CLI.

        Args:
            api_url: Optional API URL. If None, uses environment variable or default.
        """
        self.api_client = RAGAPIClient(base_url=api_url)

    def print_banner(self):
        """Print welcome banner."""
        print("=" * 70)
        print("Wikipedia RAG Question Answering System")
        print("=" * 70)
        print("Ask questions and get answers based on Wikipedia articles.")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 70)
        print()

    def print_result(self, result):
        """Print RAG result in a formatted way.

        Args:
            result: RAGResult object
        """
        print("\n" + "─" * 70)
        print("ANSWER:")
        print("─" * 70)
        print(result.answer)

        if result.sources:
            print("\n" + "─" * 70)
            print("SOURCES:")
            print("─" * 70)
            for source in result.sources:
                print(f"[{source['number']}] {source['title']}")
                print(f"    {source['url']}")

        print("─" * 70)

    def run(self):
        """Run the interactive CLI."""
        self.print_banner()

        # Check server health on startup
        try:
            print("Checking server connection...")
            health = self.api_client.check_health()
            if health.status == "healthy":
                print("✓ Connected to server")
            else:
                print(f"⚠ Server status: {health.status}")
                if health.message:
                    print(f"  {health.message}")
            print()
        except ServerNotReachableError as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease start the server first:")
            print("  nix run .#server")
            print("\nOr set AISHE_API_URL to point to a running server.")
            sys.exit(1)
        except Exception as e:
            print(f"\n⚠ Warning: Could not check server health: {e}")
            print("Continuing anyway...\n")

        while True:
            try:
                # Get user input
                question = input("\nYour question: ").strip()

                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                # Skip empty questions
                if not question:
                    continue

                # Process question
                print("\nSearching Wikipedia and generating answer...")
                result = self.api_client.ask_question(question)

                # Display result
                self.print_result(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except ServerNotReachableError as e:
                print(f"\n❌ Server Error: {e}")
                print("\nThe server may have stopped. Please restart it:")
                print("  nix run .#server")
            except ServerError as e:
                print(f"\n❌ Server Error: {e}")
            except APIClientError as e:
                print(f"\n❌ Error: {e}")
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Entry point for the CLI."""
    cli = RAGCLI()
    cli.run()


if __name__ == "__main__":
    main()
