"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using Tavily API."""
        import os
        from tavily import TavilyClient
        from multi_agent_research_lab.core.config import get_settings
        
        settings = get_settings()
        api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            print("[SearchClient] CẢNH BÁO: Không tìm thấy TAVILY_API_KEY. Trả về kết quả rỗng.")
            return []
            
        client = TavilyClient(api_key=api_key)
        
        print(f"[SearchClient] Gọi Tavily API tìm kiếm: '{query}'...")
        response = client.search(query=query, search_depth="basic", max_results=max_results)
        
        results = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    metadata={"source": "tavily", "score": item.get("score", 0.0)}
                )
            )
        return results
