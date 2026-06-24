"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        llm = LLMClient()
        system_prompt = "You are a Writer. Synthesize a final answer based on research and analysis."
        user_prompt = f"Query: {state.request.query}\nResearch: {state.research_notes}\nAnalysis: {state.analysis_notes}"
        
        response = llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content
        return state
