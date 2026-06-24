"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        
        if state.iteration >= 5: # Max iterations
            state.record_route("__end__")
            return state

        llm = LLMClient()
        system_prompt = (
            "You are a Supervisor in a research team. Your job is to decide which worker should run next based on the current state.\n"
            "Rules:\n"
            "1. If 'sources' is 0, reply EXACTLY with: researcher\n"
            "2. If 'research_notes' has content but 'analysis_notes' is None, reply EXACTLY with: analyst\n"
            "3. If 'analysis_notes' has content but 'final_answer' is None, reply EXACTLY with: writer\n"
            "4. If 'final_answer' has content, reply EXACTLY with: __end__\n"
            "Only output the exact word (researcher, analyst, writer, or __end__). No other text."
        )
        user_prompt = f"Current state:\n- sources: {len(state.sources)}\n- research_notes: {state.research_notes}\n- analysis_notes: {state.analysis_notes}\n- final_answer: {state.final_answer}"
        
        response = llm.complete(system_prompt, user_prompt)
        next_route = response.content.strip()
        if next_route not in ["researcher", "analyst", "writer", "__end__"]:
             next_route = "__end__"
        
        state.record_route(next_route)
        return state
