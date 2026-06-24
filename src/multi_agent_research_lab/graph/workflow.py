"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph."""
        from langgraph.graph import StateGraph, START, END
        from multi_agent_research_lab.agents.supervisor import SupervisorAgent
        from multi_agent_research_lab.agents.researcher import ResearcherAgent
        from multi_agent_research_lab.agents.analyst import AnalystAgent
        from multi_agent_research_lab.agents.writer import WriterAgent

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        graph = StateGraph(ResearchState)

        graph.add_node("supervisor", supervisor.run)
        graph.add_node("researcher", researcher.run)
        graph.add_node("analyst", analyst.run)
        graph.add_node("writer", writer.run)

        graph.add_edge(START, "supervisor")

        def route_condition(state: ResearchState) -> str:
            if not state.route_history:
                return END
            last_route = state.route_history[-1]
            return END if last_route == "__end__" else last_route

        graph.add_conditional_edges("supervisor", route_condition, {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        })

        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        app = self.build()
        final_state_dict = app.invoke(state)
        # In this implementation, the state is passed by reference, 
        # so the object 'final_state_dict' is the same or a dict representation.
        if isinstance(final_state_dict, ResearchState):
            return final_state_dict
        return ResearchState(**final_state_dict)
