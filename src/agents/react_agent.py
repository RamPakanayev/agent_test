from typing import Dict, List, Optional
from pydantic import BaseModel
from rag.rag_system import RAGSystem
from graph.knowledge_graph import KnowledgeGraph
from openai import OpenAI

class Thought(BaseModel):
    """Represents a thought or reasoning step in the agent's decision-making process."""
    content: str  # The content of the thought
    confidence: float  # How confident the agent is in this thought (0-1)

class Action(BaseModel):
    """Represents an action that the agent can take."""
    type: str  # The type of action (e.g., "search_graph", "retrieve_documents")
    parameters: Dict[str, str]  # Parameters for the action

class Observation(BaseModel):
    """Represents the result of an action taken by the agent."""
    content: str  # The content of the observation
    success: bool  # Whether the action was successful

class ReActAgent:
    """
    ReAct Agent Implementation

    This agent implements the ReAct (Reasoning + Acting) pattern, which is a framework for building
    agents that can reason about problems and take actions to solve them. The pattern consists of
    three main components:

    1. Think: The agent reasons about the current situation and decides what to do next
    2. Act: The agent takes an action based on its reasoning
    3. Observe: The agent observes the results of its action and updates its understanding

    The agent maintains a history of its thoughts, actions, and observations, which can be used
    to understand how it arrived at its conclusions.
    """
    
    def __init__(self, rag_system: RAGSystem, max_steps: int = 5):
        """
        Initialize the ReAct agent.
        
        Args:
            rag_system: The RAG system to use for document retrieval
            max_steps: Maximum number of reasoning steps to take
        """
        self.rag_system = rag_system
        self.knowledge_graph = rag_system.knowledge_graph
        self.max_steps = max_steps
        self.history: List[Dict] = []
        self.client = OpenAI()
        
    def _think(self, query: str, context: List[Dict], step: int) -> Thought:
        """Generate a thought based on the current state, including entity extraction."""
        try:
            # Use OpenAI to generate a thought
            print(f"\n{'═'*60}")
            print(f"🧠 STEP {step+1}: THINKING")
            print(f"{'═'*60}")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing information. Generate a thought about what to do next, including identifying entities or keywords from the query for graph search."},
                    {"role": "user", "content": f"Query: {query}\nContext: {context}\nWhat should I think about next? Identify any entities or keywords in the query that could be used to search a knowledge graph."}
                ],
                temperature=0.7,
                max_tokens=150
            )
            thought_content = response.choices[0].message.content
            print(f"\n🤔 Thought: {thought_content}")
            return Thought(content=thought_content, confidence=0.8)
        except Exception as e:
            print(f"Error generating thought: {e}")
            fallback_thought = "I should analyze the available information directly."
            print(f"🤔 Fallback Thought: {fallback_thought}")
            return Thought(
                content=fallback_thought,
                confidence=0.6
            )
        
    def _act(self, thought: Thought, step: int) -> Action:
        """Determine the next action based on the thought, deciding between graph search and vector retrieval."""
        try:
            print(f"\n{'─'*60}")
            print(f"🧠 STEP {step+1}: ACTING")
            print(f"{'─'*60}")
            
            # Use OpenAI to decide on an action
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant deciding what action to take. Choose between 'search_graph' for structured knowledge graph traversal or 'retrieve_documents' for vector similarity search. Use 'search_graph' when entities or relationships are mentioned, and 'retrieve_documents' when broader context or detailed text is needed."},
                    {"role": "user", "content": f"Based on this thought: {thought.content}\nWhat action should I take?"}
                ],
                temperature=0.7,
                max_tokens=100
            )
            action_content = response.choices[0].message.content.lower()
            
            if 'graph' in action_content or 'search_graph' in action_content or 'entities' in action_content or 'relationship' in action_content:
                action = Action(type="search_graph", parameters={"query": thought.content})
            else:
                action = Action(type="retrieve_documents", parameters={"query": thought.content})
                
            print(f"🔍 Action: {action.type} - {action.parameters}")
            return action
        except Exception as e:
            print(f"Error deciding action: {e}")
            fallback_action = Action(
                type="search_graph",
                parameters={"query": thought.content}
            )
            print(f"🔍 Fallback Action: {fallback_action.type} - {fallback_action.parameters}")
            return fallback_action
        
    def _observe(self, action: Action, step: int) -> Observation:
        """Execute the action and observe the result, using graph context to filter vector searches if needed."""
        try:
            print(f"\n{'─'*60}")
            print(f"🧠 STEP {step+1}: OBSERVING")
            print(f"{'─'*60}")
            relevance_threshold = 1.0  # You can tune this value
            if action.type == "search_graph":
                # Try to search the knowledge graph
                nodes = self.knowledge_graph.semantic_search(
                    action.parameters["query"]
                )
                observation_content = str([node.properties for node in nodes]) if nodes else "No relevant nodes found"
                success = bool(nodes)
            elif action.type == "retrieve_documents":
                # Try to retrieve documents with relevance threshold, using graph context if available
                docs = self.rag_system.retrieve_with_graph(
                    action.parameters["query"], top_k=5
                )
                observation_content = str([doc.get('content', doc) for doc in docs]) if docs else "No relevant documents found"
                success = bool(docs)
            else:
                observation_content = "Unknown action type"
                success = False
                
            # Truncate long observations for display
            display_content = (observation_content[:300] + "...") if len(observation_content) > 300 else observation_content
            print(f"👀 Observation: {display_content}")
            
            observation = Observation(
                content=observation_content,
                success=success
            )
            return observation
        except Exception as e:
            print(f"Error during observation: {e}")
            fallback_observation = Observation(
                content=f"Error performing action: {str(e)}",
                success=False
            )
            print(f"👀 Fallback Observation: Error performing action")
            return fallback_observation
            
    def _direct_response(self, query: str) -> str:
        """Generate a direct response when RAG/embeddings fail."""
        try:
            print("\n⚠️ Using direct response mode (RAG/KG not used)")
            
            # Use OpenAI to generate a direct response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a knowledgeable AI assistant specializing in artificial intelligence and machine learning. 
                    Provide clear, accurate, and helpful responses about AI concepts and their relationships."""},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating direct response: {e}")
            return f"I apologize, but I'm having trouble generating a response at the moment. Error: {str(e)}"
            
    def process_query(self, query: str) -> str:
        """
        Process a query using the ReAct pattern.
        
        This is the main method that implements the ReAct loop:
        1. Think about what to do
        2. Take an action
        3. Observe the results
        4. Repeat until done
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response to the query
        """
        self.history = []
        context = self.rag_system.retrieve_with_graph(query)

        # 1. Always do a THINK step first
        thought = self._think(query, context, step=0)
        self.history.append({"step": 0, "thought": thought.dict()})

        # 2. ACT based on that thought
        action = self._act(thought, 0)
        self.history.append({"step": 0, "action": action.dict()})

        # 3. OBSERVE the result
        observation = self._observe(action, 0)
        self.history.append({"step": 0, "observation": observation.dict()})

        # 4. Update context and *then* decide if you can answer immediately
        if observation.success:
            context.append({"type":"observation","content":observation.content})
        response = self.rag_system.generate_response(query, context)
        if response.strip() != "I Don't Know":
            return response

        # 5. Otherwise continue your full loop from step 1→max_steps…
        for step in range(1, self.max_steps):
            try:
                # Think
                thought = self._think(query, context, step)
                self.history.append({
                    "step": step,
                    "thought": thought.dict()
                })
                
                # Act
                action = self._act(thought, step)
                self.history.append({
                    "step": step,
                    "action": action.dict()
                })
                
                # Observe
                observation = self._observe(action, step)
                self.history.append({
                    "step": step,
                    "observation": observation.dict()
                })
                
                # Update context with new information
                if observation.success:
                    context.append({
                        "type": "observation",
                        "content": observation.content,
                        "metadata": {"success": observation.success}
                    })
                    print(f"\n✅ Context updated with new information")
                else:
                    print(f"\n⚠️ No new information found")
                
                # After every step, check if the current context yields a valid answer
                print("\n🎯 Generating response from current context after step...")
                response = self.rag_system.generate_response(query, context)
                if response.strip() != "I Don't Know":
                    print("\n🏁 Early exit: Found a valid answer after step. Returning response.")
                    return response
                if response.strip() == "I Don't Know":
                    print("\n🚪 Early exit: The system cannot answer. Returning 'I Don't Know'.")
                    return response
                
                # Check if we have enough information
                if observation.success and thought.confidence > 0.9:
                    print(f"\n✅ Sufficient information found after {step+1} steps")
                    break
                
            except Exception as e:
                print(f"Error in reasoning step {step}: {e}")
                continue
        
        # Generate final response
        print("\n🎯 Generating final response...")
        try:
            response = self.rag_system.generate_response(query, context)
            if not response or "error" in response.lower():
                print("⚠️ Error in generated response, returning 'I Don't Know'")
                return "I Don't Know"
            print("✅ Final response generated successfully")
            return response
        except Exception as e:
            print(f"Error generating final response: {e}")
            return "I Don't Know"
        
    def get_reasoning_chain(self) -> List[Dict]:
        """
        Get the complete reasoning chain for the last query.
        
        This returns the history of thoughts, actions, and observations
        that led to the final response. This is useful for understanding
        how the agent arrived at its conclusions.
        
        Returns:
            A list of dictionaries containing the reasoning chain
        """
        return self.history 