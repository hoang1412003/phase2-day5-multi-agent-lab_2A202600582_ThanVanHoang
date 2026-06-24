"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.services.search_client import SearchClient
        
        search = SearchClient()
        llm = LLMClient()
        
        # 1. Search
        docs = search.search(state.request.query, state.request.max_sources)
        state.sources.extend(docs)
        
        # 2. Extract and summarize notes
        system_prompt = "You are a Researcher. Summarize the following sources."
        user_prompt = f"Query: {state.request.query}\nSources: {[d.snippet for d in docs]}"
        response = llm.complete(system_prompt, user_prompt)
        
        state.research_notes = response.content
        return state
