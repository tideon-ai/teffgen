"""
Core Agent implementation for effGen.

The main Agent class with:
- ReAct loop (Reason + Act)
- Tool selection and execution
- Sub-agent integration via router
- Memory management
- Streaming support
- State persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..memory.long_term import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryType,
    SQLiteStorageBackend,
)
from ..memory.short_term import MessageRole, ShortTermMemory
from ..models.base import BaseModel, GenerationConfig
from ..models.model_loader import ModelLoader
from ..prompts.agent_system_prompt import AgentSystemPromptBuilder
from ..prompts.tool_prompt_generator import ToolPromptGenerator
from ..tools.base_tool import BaseTool
from ..tools.fallback import ToolFallbackChain
from ..utils.circuit_breaker import CircuitBreaker
from .execution_tracker import EventType, ExecutionEvent, ExecutionTracker
from .router import RoutingDecision, RoutingStrategy, SubAgentRouter
from .state import AgentState
from .sub_agent_manager import SubAgentManager

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent execution modes."""
    SINGLE = "single"  # Single agent execution
    SUB_AGENTS = "sub_agents"  # Use sub-agents for complex tasks
    AUTO = "auto"  # Automatically decide based on router


@dataclass
class AgentConfig:
    """
    Agent configuration.

    Attributes:
        name: Agent name/identifier
        model: Model instance or name
        tools: List of available tools
        system_prompt: System-level instructions
        max_iterations: Maximum tool-use loop iterations
        temperature: Generation temperature
        enable_sub_agents: Enable sub-agent spawning
        enable_memory: Enable memory systems
        enable_streaming: Enable response streaming
        max_context_length: Maximum context window
        router_config: Configuration for sub-agent router
        sub_agent_config: Configuration for sub-agent manager
        model_config: Optional model engine configuration
        require_model: Whether model loading is required (raise error on failure)
    """
    name: str
    model: BaseModel | str
    tools: list[BaseTool] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    temperature: float = 0.7
    enable_sub_agents: bool = True
    enable_memory: bool = True
    enable_streaming: bool = False
    max_context_length: int | None = None
    router_config: dict[str, Any] = field(default_factory=dict)
    sub_agent_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] | None = None
    require_model: bool = False
    system_prompt_template: str | None = None
    verbose_tools: bool | None = None
    fallback_chain: dict[str, list] | None = None
    enable_fallback: bool = True
    memory_config: dict[str, Any] = field(default_factory=lambda: {
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 100,
        "long_term_backend": "sqlite",
        "long_term_persist_path": None,
        "auto_summarize": True,
    })


@dataclass
class AgentResponse:
    """
    Response from agent execution.

    Attributes:
        output: Final output text
        success: Whether execution succeeded
        mode: Execution mode used
        iterations: Number of iterations performed
        tool_calls: Number of tool calls made
        tokens_used: Total tokens consumed
        execution_time: Time taken in seconds
        execution_trace: Full execution trace
        execution_tree: Hierarchical execution tree
        routing_decision: Routing decision (if sub-agents used)
        metadata: Additional metadata
    """
    output: str
    success: bool = True
    mode: AgentMode = AgentMode.SINGLE
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    execution_time: float = 0.0
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    execution_tree: dict[str, Any] = field(default_factory=dict)
    routing_decision: RoutingDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "mode": self.mode.value,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "execution_time": round(self.execution_time, 2),
            "execution_trace": self.execution_trace,
            "execution_tree": self.execution_tree,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "metadata": self.metadata
        }


class Agent:
    """
    Main Agent implementation with ReAct loop and sub-agent support.

    The agent can:
    - Execute tasks using ReAct (Reason + Act) pattern
    - Intelligently spawn sub-agents for complex tasks
    - Use tools to interact with external systems
    - Manage conversation memory
    - Stream responses
    - Save/load state
    """

    # Default ReAct prompt template
    REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that can reason step-by-step and use tools.
{conversation_history}
Available tools:
{tools_description}

IMPORTANT: If there is previous conversation context above, use that information to answer questions about past interactions.

Use the following format:

Question: the input question or task
Thought: think step-by-step about what to do next
Action: the tool to use (or "Final Answer" when ready to respond)
Action Input: the input for the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete response to the original question

Begin!

Question: {task}
{scratchpad}"""

    def __init__(self, config: AgentConfig):
        """
        Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name

        # Model initialization
        self.model_loader = ModelLoader()
        if isinstance(config.model, BaseModel):
            # Model instance provided directly
            self.model = config.model
            self.model_name = getattr(config.model, 'model_name', 'custom')
        elif isinstance(config.model, str):
            # Model name provided - load it
            self.model_name = config.model
            try:
                logger.info(f"Loading model: {self.model_name}")
                self.model = self.model_loader.load_model(
                    self.model_name,
                    engine_config=config.model_config
                )
                logger.info(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model '{self.model_name}': {e}")
                self.model = None
                if config.require_model:
                    raise RuntimeError(f"Failed to load required model: {e}")
        else:
            # No model provided
            self.model_name = None
            self.model = None

        # Tools
        self.tools = {tool.name: tool for tool in config.tools}

        # Tool prompt generator for enhanced ReAct prompts
        self._tool_prompt_generator = ToolPromptGenerator(
            tools=config.tools,
            model_name=self.model_name or "",
        )

        # Determine verbose_tools setting: auto-detect from model size if not set
        if config.verbose_tools is not None:
            self._verbose_tools = config.verbose_tools
        else:
            self._verbose_tools = self._auto_detect_verbose()

        # Tool fallback chain
        self._fallback_chain = ToolFallbackChain(
            custom_chains=config.fallback_chain
        )
        self._enable_fallback = config.enable_fallback

        # Circuit breaker for tool failures
        self._circuit_breaker = CircuitBreaker()

        # Auto-generate system prompt if tools are present and using default prompt
        self._system_prompt_builder = AgentSystemPromptBuilder(
            model_name=self.model_name or "",
        )
        if config.tools and config.system_prompt == "You are a helpful AI assistant.":
            self.config.system_prompt = self._build_system_prompt()

        # State management
        self.state = AgentState(agent_id=self.name)

        # Sub-agent components
        self.router = None
        self.sub_agent_manager = None
        if config.enable_sub_agents:
            self.router = SubAgentRouter(
                config=config.router_config,
                llm_client=self  # Pass self as LLM client
            )
            self.sub_agent_manager = SubAgentManager(
                parent_agent=self,
                config=config.sub_agent_config
            )

        # Execution tracker
        self.execution_tracker = ExecutionTracker()

        # Memory system
        mem_cfg = config.memory_config or {}
        stm_max_tokens = mem_cfg.get("short_term_max_tokens", 4096)
        stm_max_messages = mem_cfg.get("short_term_max_messages", 100)
        self.short_term_memory = ShortTermMemory(
            max_tokens=stm_max_tokens,
            max_messages=stm_max_messages,
            summarization_threshold=mem_cfg.get("summarization_threshold", 0.8),
            keep_recent_messages=mem_cfg.get("keep_recent_messages", 4),
        )

        # Long-term memory (optional, requires persist path)
        self.long_term_memory: LongTermMemory | None = None
        if config.enable_memory:
            persist_path = mem_cfg.get("long_term_persist_path")
            if persist_path:
                import os
                persist_path = os.path.expanduser(persist_path)
                backend_type = mem_cfg.get("long_term_backend", "sqlite")
                if backend_type == "sqlite":
                    backend = SQLiteStorageBackend(
                        os.path.join(persist_path, "long_term.db")
                    )
                else:
                    backend = JSONStorageBackend(
                        os.path.join(persist_path, "long_term.json")
                    )
                self.long_term_memory = LongTermMemory(backend=backend)
                self.long_term_memory.start_session(name=self.name)

    def _build_system_prompt(self) -> str:
        """Build a dynamic system prompt based on agent configuration and tools."""
        return self._system_prompt_builder.build(
            tools=self.config.tools,
            agent_name=self.name,
            base_system_prompt=None,  # Will generate default role
            enable_fallback=self._enable_fallback,
            verbose=self._verbose_tools,
        )

    def _auto_detect_verbose(self) -> bool:
        """Auto-detect whether to use verbose tool descriptions based on model size."""
        name = (self.model_name or "").lower()
        # Check for known small models (< 3B) -> full verbose with examples
        # Check for medium models (3B-7B) -> verbose without examples
        # Check for large models (> 7B) or API models -> compact
        for indicator in ["0.5b", "1b", "1.5b", "2b"]:
            if indicator in name:
                return True
        for indicator in ["3b", "4b", "5b", "7b"]:
            if indicator in name:
                return True
        # API models
        for indicator in ["gpt", "claude", "gemini"]:
            if indicator in name:
                return False
        # Default: verbose (safe for SLMs)
        return True

    def run(self,
            task: str,
            mode: AgentMode = AgentMode.AUTO,
            context: dict[str, Any] | None = None,
            **kwargs) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: Task description
            mode: Execution mode (single, sub_agents, auto)
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            AgentResponse with results
        """
        start_time = time.time()
        context = context or {}

        # Track task start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_START,
            agent_id=self.name,
            message=f"Starting task: {task[:100]}...",
            data={"task": task, "mode": mode.value}
        ))

        try:
            # Determine execution mode
            if mode == AgentMode.AUTO and self.config.enable_sub_agents:
                # Use router to decide
                routing_decision = self.router.route(task, context)

                if routing_decision.use_sub_agents:
                    response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                else:
                    response = self._run_single_agent(task, context, **kwargs)
            elif mode == AgentMode.SUB_AGENTS and self.config.enable_sub_agents:
                # Force sub-agent mode
                routing_decision = self.router.route(task, context)
                response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
            else:
                # Single agent mode
                response = self._run_single_agent(task, context, **kwargs)

            # Add execution metadata
            response.execution_time = time.time() - start_time
            response.execution_trace = self.execution_tracker.get_trace()
            response.execution_tree = self.execution_tracker.generate_execution_tree()

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_COMPLETE,
                agent_id=self.name,
                message=f"Task completed in {response.execution_time:.2f}s",
                data={
                    "execution_time": response.execution_time,
                    "tokens_used": response.tokens_used,
                    "tool_calls": response.tool_calls
                }
            ))

            # Store conversation in short-term memory for context retention
            if response.success and response.output:
                self.short_term_memory.add_user_message(task)
                self.short_term_memory.add_assistant_message(response.output)
                logger.debug(
                    f"Stored conversation turn in memory "
                    f"(total: {self.short_term_memory.total_messages_added} messages)"
                )

                # Persist important facts to long-term memory if available
                if self.long_term_memory and response.tool_calls > 0:
                    self.long_term_memory.add_memory(
                        content=f"Q: {task}\nA: {response.output}",
                        memory_type=MemoryType.CONVERSATION,
                        importance=ImportanceLevel.MEDIUM,
                        tags=["conversation"],
                    )

            return response

        except Exception as e:
            # Track failure
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_FAILED,
                agent_id=self.name,
                message=f"Task failed: {str(e)}",
                data={"error": str(e)}
            ))

            return AgentResponse(
                output=f"Error: {str(e)}",
                success=False,
                execution_time=time.time() - start_time,
                execution_trace=self.execution_tracker.get_trace(),
                metadata={"error": str(e)}
            )

    def _run_single_agent(self,
                         task: str,
                         context: dict[str, Any],
                         **kwargs) -> AgentResponse:
        """
        Execute task using single agent with ReAct loop or direct inference.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        iterations = 0
        tool_calls = 0
        tokens_used = 0
        scratchpad = ""
        max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop
        previous_actions: list[tuple[str, str]] = []  # Track (action, input) pairs for loop detection
        while iterations < max_iterations:
            iterations += 1

            # Build prompt using ToolPromptGenerator or custom template
            if self.config.system_prompt_template:
                # User-provided custom template
                tools_description = self._get_tools_description()
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_description,
                    conversation_history=conversation_history,
                    task=task,
                    scratchpad=scratchpad
                )
            else:
                # Use enhanced ToolPromptGenerator
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                )

            # Debug: log first iteration prompt to see if history is included
            if iterations == 1 and conversation_history:
                logger.info(f"[Memory] Including conversation history ({len(self.short_term_memory.messages)} messages)")

            # Track reasoning step
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.REASONING_STEP,
                agent_id=self.name,
                message=f"Iteration {iterations}: Reasoning...",
                data={"iteration": iterations}
            ))

            # Generate response
            response = self._generate(prompt, **kwargs)
            tokens_used += response.get("tokens_used", 0)

            # Debug: Log the raw response
            logger.info(f"[Iteration {iterations}] Raw model output: {response['text'][:300]}...")
            logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

            # Parse response
            parsed = self._parse_react_response(response["text"])

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad
            scratchpad += f"\nThought: {parsed.get('thought', '')}"

            # Check for final answer
            if parsed.get("final_answer"):
                return AgentResponse(
                    output=parsed["final_answer"],
                    success=True,
                    mode=AgentMode.SINGLE,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens_used=tokens_used
                )

            # Check if model is stating an answer without "Final Answer:" keyword
            # This happens when model provides result after tool execution
            if tool_calls > 0 and not parsed.get("action"):
                # No action and we've used tools - model might be stating the answer
                response_text = response["text"].strip()
                # Check for answer-like patterns
                if any(phrase in response_text.lower() for phrase in ["the answer is", "the result is", "the sum is", "equals", "="]):
                    logger.info("Detected answer statement without 'Final Answer:' keyword")
                    return AgentResponse(
                        output=response_text,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=iterations,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used
                    )

            # Execute action if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Loop detection: check if we've seen this exact (action, input) before
                # Also detect fuzzy loops: same tool called 3+ times with different inputs
                # (SLMs like Llama produce slightly different formatting each time)
                current_pair = (action, action_input)
                action_call_count = sum(1 for a, _ in previous_actions if a == action)
                is_exact_loop = current_pair in previous_actions and action in self.tools
                is_fuzzy_loop = action_call_count >= 5 and action in self.tools
                if is_exact_loop or is_fuzzy_loop:
                    loop_type = "exact" if is_exact_loop else f"fuzzy ({action_call_count + 1} calls)"
                    logger.info(
                        f"[Loop detected] Repeated action '{action}' ({loop_type}) — "
                        f"breaking loop and returning last observation"
                    )
                    # Extract the last successful observation from scratchpad
                    partial = self._extract_partial_answer(scratchpad)
                    if partial:
                        return AgentResponse(
                            output=partial,
                            success=True,
                            mode=AgentMode.SINGLE,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            metadata={"reason": "loop_detected", "repeated_action": action}
                        )
                    # If no partial answer, add a hint to the scratchpad and continue
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        "\nObservation: You already computed this. "
                        "Please provide your final response using 'Final Answer:' now."
                    )
                    continue

                previous_actions.append(current_pair)

                # Check if tool is available (handle no-tool mode gracefully)
                if not self.tools or action not in self.tools:
                    # No tools available - model is hallucinating tools
                    # Guide it to provide direct answer
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += "\nObservation: No tools available. Please provide your answer directly using 'Final Answer:'."
                else:
                    # Execute tool
                    tool_result = self._execute_tool(action, action_input)
                    tool_calls += 1

                    # Add observation to scratchpad
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Log the observation for debugging
                    logger.info(f"Tool result added to scratchpad: {tool_result[:100]}...")

            else:
                # No action specified, prompt to continue
                scratchpad += "\nAction: (continue reasoning)"

        # Max iterations reached — try to extract partial answer from scratchpad
        partial_answer = self._extract_partial_answer(scratchpad)
        if partial_answer:
            logger.info("Max iterations reached, returning partial answer from scratchpad")
            return AgentResponse(
                output=partial_answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                metadata={"reason": "max_iterations_partial", "partial": True}
            )

        return AgentResponse(
            output="Maximum iterations reached without final answer.",
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            metadata={"reason": "max_iterations_reached"}
        )

    def _extract_partial_answer(self, scratchpad: str) -> str | None:
        """
        Extract the best partial answer from the scratchpad when max iterations is reached.

        Looks for patterns like "I now know the answer", recent observations with
        answer-like content, or the last substantive thought.

        Args:
            scratchpad: The accumulated scratchpad text.

        Returns:
            A partial answer string, or None if nothing useful found.
        """
        if not scratchpad:
            return None

        # Pattern 1: "I now know" type thoughts
        know_match = re.search(
            r"Thought:\s*I (?:now )?know[^.]*\.\s*(.+?)(?=\nThought:|\nAction:|\Z)",
            scratchpad, re.IGNORECASE | re.DOTALL
        )
        if know_match:
            return know_match.group(1).strip()

        # Pattern 2: Last observation with a clear result value
        observations = re.findall(r"Observation:\s*(.+?)(?=\nThought:|\nAction:|\Z)", scratchpad, re.DOTALL)
        if observations:
            last_obs = observations[-1].strip()
            # If last observation looks like a useful result (not an error)
            if last_obs and not last_obs.lower().startswith("error"):
                return last_obs

        # Pattern 3: Last substantive thought
        thoughts = re.findall(r"Thought:\s*(.+?)(?=\nAction:|\nObservation:|\Z)", scratchpad, re.DOTALL)
        if thoughts:
            last_thought = thoughts[-1].strip()
            if len(last_thought) > 20:
                return last_thought

        return None

    def _run_with_sub_agents(self,
                            task: str,
                            routing_decision: RoutingDecision,
                            context: dict[str, Any],
                            **kwargs) -> AgentResponse:
        """
        Execute task using sub-agents based on routing decision.

        Args:
            task: Task description
            routing_decision: Router's decision
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Track decomposition
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_DECOMPOSITION,
            agent_id=self.name,
            message=f"Decomposed into {routing_decision.num_sub_agents} subtasks using {routing_decision.strategy.value}",
            data={
                "strategy": routing_decision.strategy.value,
                "num_subtasks": routing_decision.num_sub_agents,
                "specializations": routing_decision.specializations
            }
        ))

        # Execute based on strategy
        strategy = routing_decision.strategy
        subtasks = routing_decision.decomposition

        if strategy == RoutingStrategy.PARALLEL_SUB_AGENTS:
            # Execute in parallel
            results = asyncio.run(
                self.sub_agent_manager.execute_parallel(subtasks)
            )
        elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
            # Execute sequentially
            results = self.sub_agent_manager.execute_sequential(subtasks)
        elif strategy == RoutingStrategy.HYBRID:
            # Execute with hybrid approach
            results = self.sub_agent_manager.execute_hybrid(subtasks)
        else:
            # Default to sequential
            results = self.sub_agent_manager.execute_sequential(subtasks)

        # Synthesize results
        synthesis = self.sub_agent_manager.synthesize_results(
            results,
            task,
            strategy
        )

        # Calculate totals
        total_tokens = synthesis["metrics"]["total_tokens_used"]
        total_tool_calls = synthesis["metrics"]["total_tool_calls"]

        return AgentResponse(
            output=synthesis["final_output"],
            success=synthesis["successful"] > 0,
            mode=AgentMode.SUB_AGENTS,
            iterations=len(subtasks),
            tool_calls=total_tool_calls,
            tokens_used=total_tokens,
            routing_decision=routing_decision,
            metadata={
                "synthesis": synthesis,
                "failed_subtasks": synthesis["failed"]
            }
        )

    def _generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Generate response from model with retry logic for empty responses.

        Retries up to 3 times on empty responses with exponential backoff
        and slightly increasing temperature.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'text', 'tokens_used', and other metadata

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        max_retries = 3
        backoff_delays = [0.5, 1.0, 2.0]
        base_temperature = kwargs.get('temperature', self.config.temperature)

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
            "\n\n\n"
        ]

        last_error = None
        total_tokens = 0

        for attempt in range(max_retries):
            try:
                # Slightly increase temperature on retries to get different output
                retry_temperature = min(base_temperature + (attempt * 0.1), 1.0)

                gen_config = GenerationConfig(
                    temperature=retry_temperature,
                    max_tokens=kwargs.get('max_tokens', 1024),
                    top_p=kwargs.get('top_p', 0.9),
                    stop_sequences=kwargs.get('stop_sequences', default_stop_sequences)
                )

                result = self.model.generate(prompt, config=gen_config)

                response_text = result.text if result and result.text else ""
                tokens_used = result.tokens_used if result and hasattr(result, 'tokens_used') else 0
                finish_reason = result.finish_reason if result and hasattr(result, 'finish_reason') else "unknown"
                total_tokens += tokens_used

                # If we got a non-empty response, return it
                if response_text.strip():
                    return {
                        "text": response_text,
                        "tokens_used": total_tokens,
                        "finish_reason": finish_reason,
                        "metadata": result.metadata if result and hasattr(result, 'metadata') else {}
                    }

                # Empty response — retry
                if attempt < max_retries - 1:
                    logger.info(
                        f"Empty response on attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {backoff_delays[attempt]}s with temperature={retry_temperature:.2f}"
                    )
                    time.sleep(backoff_delays[attempt])
                else:
                    logger.warning(f"Empty response after {max_retries} attempts")

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Generation error on attempt {attempt + 1}/{max_retries}: {e}, "
                        f"retrying in {backoff_delays[attempt]}s"
                    )
                    time.sleep(backoff_delays[attempt])
                else:
                    logger.error(f"Generation failed after {max_retries} attempts: {e}")

        # All retries exhausted
        if last_error:
            logger.warning(f"Returning empty response due to generation failure: {last_error}")
        return {
            "text": "",
            "tokens_used": total_tokens,
            "finish_reason": "error",
            "metadata": {"error": str(last_error) if last_error else "empty_response"}
        }

    def _parse_react_response(self, text: str) -> dict[str, Any]:
        """
        Parse ReAct formatted response with robust error handling.

        Args:
            text: Response text

        Returns:
            Dictionary with parsed components

        Notes:
            This parser handles various formats and edge cases:
            - Case-insensitive matching
            - Multiple thought/action patterns
            - Malformed responses
            - Missing fields
        """
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return parsed

        try:
            # Check for final answer first (highest priority)
            # NOTE: "Answer:" must be at the start of a line to avoid greedy
            # mid-text matches (e.g. "The answer is 42" should NOT match here).
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)"
            ]

            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        # Stop at next section marker or observation
                        answer = re.split(r'\n(?:Question|Thought|Action):', answer, maxsplit=1)[0].strip()
                        # Strip trailing unrelated content (e.g. Phi-4 generating
                        # follow-up questions after the answer like "...is 42.What year...")
                        # Match sentence boundary (.!?) followed by a new sentence
                        # that itself contains a question mark — likely a hallucinated follow-up.
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
                        parsed["final_answer"] = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return parsed
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # Extract thought
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
            ]

            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        parsed["thought"] = thought
                        logger.debug(f"Extracted thought: {thought[:100]}...")
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # Extract action
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)"
            ]

            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        action = action_match.group(1).strip()
                        # Clean up common artifacts
                        action = action.replace('"', '').replace("'", "")

                        # Check if action is actually "Final Answer" - treat it as final answer, not tool
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            # When model writes "Action: Final Answer", the answer may be:
                            # 1. On the same line: "Action: Final Answer: The answer is 42"
                            # 2. On the Action Input line: "Action Input: The answer is 42"
                            # But NOT if Action Input is JSON (model repeating tool input)

                            # Try same-line first — [: \t]+ excludes newlines
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)", text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action line")
                                    return parsed

                            # Try Action Input line (only if it's natural language, not JSON)
                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action Input line")
                                    return parsed

                            # If we get here, the model wrote "Action: Final Answer"
                            # but didn't provide a proper answer text. Don't extract
                            # anything — let the loop continue for another iteration.
                            logger.debug("Action: Final Answer detected but no answer text found")
                            break

                        # Handle function-call format: tool_name(args) or tool_name("args")
                        # Extract just the tool name and put args into action_input
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip()
                            # Remove surrounding quotes if present
                            embedded_args = embedded_args.strip('"\'')
                            parsed["action"] = tool_name
                            # Only set action_input if not already set
                            if "action_input" not in parsed or not parsed["action_input"]:
                                parsed["action_input"] = embedded_args
                            logger.debug(f"Extracted function-call style: action={tool_name}, input={embedded_args[:100]}...")
                        else:
                            parsed["action"] = action
                            logger.debug(f"Extracted action: {action}")
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # Extract action input (only if not already set from function-call style)
            # Skip if we already have embedded args from tool_name(args) format
            if "action_input" not in parsed or not parsed.get("action_input"):
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)"
                ]

                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            # Remove trailing observation text if present
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            parsed["action_input"] = action_input
                            logger.debug(f"Extracted action input: {action_input[:100]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in parse_react_response: {e}", exc_info=True)
            # Return partial parse results even if there was an error

        return parsed

    @staticmethod
    def _run_coroutine_sync(coro):
        """
        Run an async coroutine from synchronous code.

        Uses a simple strategy: try asyncio.run() first. If an event loop
        is already running, fall back to a thread-based approach.
        """
        try:
            # No running loop — simplest path
            return asyncio.run(coro)
        except RuntimeError:
            # Event loop already running — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=60)

    @staticmethod
    def _sanitize_tool_input(tool_input: str, max_length: int = 10000) -> str:
        """
        Sanitize tool input by stripping control characters and limiting length.

        Args:
            tool_input: Raw input string.
            max_length: Maximum allowed input length.

        Returns:
            Sanitized input string.
        """
        if not tool_input:
            return tool_input
        # Strip control characters except newline and tab
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', tool_input)
        # Limit length
        if len(sanitized) > max_length:
            logger.warning(f"Tool input truncated from {len(sanitized)} to {max_length} chars")
            sanitized = sanitized[:max_length]
        return sanitized

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with circuit breaker, fallback support, and input sanitization.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        # Sanitize input
        tool_input = self._sanitize_tool_input(tool_input)

        # Circuit breaker check
        if not self._circuit_breaker.is_available(tool_name):
            logger.info(f"Circuit breaker OPEN for '{tool_name}', skipping execution")
            return f"Error executing tool '{tool_name}': tool temporarily disabled due to repeated failures"

        result_str = self._execute_tool_once(tool_name, tool_input)

        # Update circuit breaker
        if result_str.startswith("Error executing tool"):
            self._circuit_breaker.record_failure(tool_name)
        else:
            self._circuit_breaker.record_success(tool_name)

        # Check if primary tool failed and fallback is enabled
        if (
            self._enable_fallback
            and result_str.startswith("Error executing tool")
            and self._fallback_chain.has_fallbacks(tool_name)
        ):
            fallbacks = self._fallback_chain.get_fallbacks(tool_name)
            for fb_name in fallbacks:
                if fb_name not in self.tools:
                    continue
                logger.info(f"Tool '{tool_name}' failed, trying fallback: {fb_name}")
                fb_result = self._execute_tool_once(fb_name, tool_input)
                if not fb_result.startswith("Error executing tool"):
                    logger.info(f"Fallback '{fb_name}' succeeded for '{tool_name}'")
                    return f"[Fallback: used {fb_name} instead of {tool_name}] {fb_result}"
            logger.info(f"All fallbacks exhausted for '{tool_name}'")

        return result_str

    def _execute_tool_once(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a single tool with robust error handling (no fallback).

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        # Track tool call start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TOOL_CALL_START,
            agent_id=self.name,
            message=f"Calling tool: {tool_name}",
            data={"tool_name": tool_name, "tool_input": tool_input}
        ))

        try:
            # Validate tool exists
            if not tool_name:
                raise ValueError("Tool name cannot be empty")

            if tool_name not in self.tools:
                available_tools = ", ".join(self.tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not available. "
                    f"Available tools: {available_tools}"
                )

            tool = self.tools[tool_name]

            # Parse input intelligently
            input_dict = {}
            if tool_input:
                try:
                    # Try parsing as JSON first
                    input_dict = json.loads(tool_input)
                    if not isinstance(input_dict, dict):
                        # JSON parsed but not a dict - need to intelligently map to tool parameters
                        input_dict = self._map_input_to_parameters(tool, input_dict)
                except json.JSONDecodeError:
                    # Not valid JSON — SLMs often produce Python-style dicts with
                    # single-quoted strings (e.g. {"data": '{"key": "val"}'}).
                    # Try ast.literal_eval as a fallback before plain-text mapping.
                    try:
                        import ast
                        parsed = ast.literal_eval(tool_input)
                        if isinstance(parsed, dict):
                            input_dict = parsed
                        else:
                            input_dict = self._map_input_to_parameters(tool, tool_input)
                    except (ValueError, SyntaxError):
                        # Not valid Python either — use plain text mapping
                        input_dict = self._map_input_to_parameters(tool, tool_input)
                except Exception as e:
                    logger.warning(f"Error parsing tool input, using as plain text: {e}")
                    input_dict = self._map_input_to_parameters(tool, tool_input)

            # Strip markdown code fences from 'code' param even after JSON parse
            if isinstance(input_dict, dict) and 'code' in input_dict and isinstance(input_dict['code'], str):
                code_val = input_dict['code']
                if '```' in code_val:
                    import re as _re
                    code_val = _re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_val, flags=_re.MULTILINE)
                    code_val = _re.sub(r'\n?```$', '', code_val, flags=_re.MULTILINE)
                    input_dict['code'] = code_val.strip()

            logger.debug(f"Executing tool '{tool_name}' with input: {input_dict}")

            # Execute tool (handle both sync and async)
            try:
                result = tool.execute(**input_dict)

                # Handle async results (coroutines) cleanly
                if asyncio.iscoroutine(result):
                    result = self._run_coroutine_sync(result)

            except TypeError as e:
                logger.error(f"Tool parameter error: {e}")
                raise ValueError(
                    f"Tool '{tool_name}' parameter error: {str(e)}. "
                    f"Input provided: {input_dict}"
                )

            # Convert result to string safely
            if result is None:
                result_str = "No result returned"
            elif hasattr(result, 'output'):
                # ToolResult object - extract output
                if hasattr(result, 'success') and not result.success:
                    error_msg = getattr(result, 'error', 'Unknown error')
                    result_str = f"Tool execution failed: {error_msg}"
                else:
                    output = result.output
                    if isinstance(output, dict):
                        # Try common result keys: result, output, data, message
                        # BUG-012 fix: PythonREPL returns {result: None, stdout: "..."}
                        # when code uses print(). Prefer stdout over a None result.
                        if 'result' in output and output['result'] is not None:
                            result_str = str(output['result'])
                        elif 'stdout' in output and output['stdout']:
                            # PythonREPL/CodeExecutor: stdout has the printed output
                            parts = []
                            parts.append(output['stdout'].rstrip())
                            if output.get('stderr'):
                                parts.append(f"stderr: {output['stderr'].rstrip()}")
                            if output.get('error'):
                                parts.append(f"Error: {output['error']}")
                            result_str = '\n'.join(parts)
                        elif 'stderr' in output and output['stderr']:
                            # CodeExecutor error: stdout empty but stderr has traceback
                            result_str = f"Error: {output['stderr'].rstrip()}"
                            if output.get('exit_code'):
                                result_str += f"\n(exit code: {output['exit_code']})"
                        elif 'error' in output and output['error']:
                            result_str = f"Error: {output['error']}"
                        elif 'exit_code' in output:
                            # CodeExecutor: ran successfully but no output
                            result_str = f"Code executed successfully (exit code {output['exit_code']})"
                        elif 'output' in output:
                            result_str = str(output['output'])
                        elif 'data' in output and 'success' in output:
                            # FileOperations-style: {success, data, message}
                            if output.get('success'):
                                result_str = str(output['data']) if output['data'] is not None else output.get('message', str(output))
                            else:
                                result_str = f"Operation failed: {output.get('message', str(output))}"
                        elif 'data' in output:
                            result_str = str(output['data'])
                        elif 'message' in output:
                            result_str = str(output['message'])
                        else:
                            result_str = str(output)
                    else:
                        result_str = str(output)
            elif hasattr(result, 'result'):
                result_str = str(result.result)
            else:
                result_str = str(result)

            # Check if result indicates a tool-level failure
            if result_str.startswith("Tool execution failed:"):
                raise ValueError(result_str)

            # Track success
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_COMPLETE,
                agent_id=self.name,
                message=f"Tool {tool_name} completed",
                data={"tool_name": tool_name, "result": result_str[:200]}
            ))

            logger.info(f"Tool '{tool_name}' executed successfully")
            return result_str

        except ValueError as e:
            error_msg = str(e)
            logger.debug(f"Tool '{tool_name}' execution failed: {error_msg}")

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            return f"Error executing tool '{tool_name}': {error_msg}"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Tool '{tool_name}' execution failed: {error_msg}", exc_info=True)

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            return f"Error executing tool '{tool_name}': {error_msg}"

    def _map_input_to_parameters(self, tool, input_value):
        """
        Intelligently map input value to tool parameters.

        This handles cases where the model provides plain text or non-dict JSON
        and we need to map it to the tool's expected parameter names.

        Args:
            tool: The tool object with metadata
            input_value: The input value (string or other type)

        Returns:
            Dict mapping parameter names to values
        """
        # Get tool parameters
        if not hasattr(tool, 'metadata') or not hasattr(tool.metadata, 'parameters'):
            # No metadata - use generic "input"
            return {"input": input_value}

        params = tool.metadata.parameters
        required_params = [p for p in params if p.required]

        # Case 1: Single parameter tool
        if len(params) == 1:
            param_name = params[0].name
            return {param_name: input_value}

        # Case 2: Multiple parameters - need to be smarter
        # Common patterns for specific tools

        # Code executor pattern: expects "code" and "language"
        if any(p.name == "code" for p in params):
            # Clean markdown code fences from code input
            code_str = str(input_value)
            # Remove markdown code fences (```python, ```py, ```javascript, etc.)
            import re
            code_str = re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_str, flags=re.MULTILINE)
            code_str = re.sub(r'\n?```$', '', code_str, flags=re.MULTILINE)
            code_str = code_str.strip()

            result = {"code": code_str}
            # Add default language if required
            if any(p.name == "language" and p.required for p in params):
                result["language"] = "python"  # Default to Python
            return result

        # Calculator pattern: expects "expression"
        if any(p.name == "expression" for p in params):
            return {"expression": str(input_value)}

        # Python REPL pattern: expects "code"
        if any(p.name == "code" for p in required_params):
            return {"code": str(input_value)}

        # File ops pattern: expects "operation" and "path"
        if any(p.name == "operation" for p in required_params):
            import re
            input_str = str(input_value).lower()

            # Try to extract operation and path from the input
            result = {}

            # Detect operation from keywords
            operation = None
            if any(word in input_str for word in ["read", "reading", "show", "display", "cat", "get content"]):
                operation = "read"
            elif any(word in input_str for word in ["write", "writing", "create", "save"]):
                operation = "write"
            elif any(word in input_str for word in ["list", "listing", "ls", "dir", "show files"]):
                operation = "list"
            elif any(word in input_str for word in ["search", "find", "grep"]):
                operation = "search"
            elif any(word in input_str for word in ["metadata", "info", "stat"]):
                operation = "metadata"
            elif any(word in input_str for word in ["convert", "transform"]):
                operation = "convert"

            if operation:
                result["operation"] = operation
            else:
                # Default to read if unclear
                result["operation"] = "read"

            # Extract path - look for file paths or filenames
            path_str = str(input_value)

            # Remove file:/// prefix if present
            path_str = re.sub(r'file://+', '', path_str)

            # Try to find filenames with extensions (prefer last occurrence to avoid "path/to/file")
            # Look for patterns like: file.txt, /path/file.txt, ./file.txt
            path_matches = re.findall(r'[\w\-\.\/]+\.\w+', path_str)
            if path_matches:
                # Use the last match (most likely the actual filename)
                path_candidate = path_matches[-1]
                # If it contains slashes, prefer just the filename part unless it starts with / or ./
                if '/' in path_candidate and not path_candidate.startswith(('/', './')):
                    # Extract just the filename
                    result["path"] = path_candidate.split('/')[-1]
                else:
                    result["path"] = path_candidate
            else:
                # Look for any path-like string
                path_match = re.search(r'(?:file|path)[\s:=]+([^\s]+)', path_str, re.IGNORECASE)
                if path_match:
                    result["path"] = path_match.group(1)
                else:
                    # Just use the input as path (remove operation keywords)
                    cleaned_path = path_str
                    for op_word in ["read", "write", "list", "search", "metadata", "convert", "operation="]:
                        cleaned_path = cleaned_path.replace(op_word, "").replace(op_word.upper(), "")
                    result["path"] = cleaned_path.strip()

            # Clean up path - remove trailing whitespace but preserve absolute paths
            if result.get("path"):
                result["path"] = result["path"].strip()
                # Remove file:// prefix if still present after earlier cleanup
                if result["path"].startswith("file://"):
                    result["path"] = result["path"][7:]
                    # Ensure we keep at least one leading /
                    if not result["path"].startswith("/"):
                        result["path"] = "/" + result["path"]

            return result

        # Search pattern: expects "query"
        if any(p.name == "query" for p in params):
            return {"query": str(input_value)}

        # Default: Use first required parameter name, or first parameter name
        if required_params:
            return {required_params[0].name: str(input_value)}
        elif params:
            return {params[0].name: str(input_value)}
        else:
            # Fallback
            return {"input": str(input_value)}

    def _run_direct_inference(self,
                               task: str,
                               context: dict[str, Any],
                               **kwargs) -> AgentResponse:
        """
        Run direct inference without ReAct loop (for when no tools are available).

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Include conversation history from short-term memory for multi-turn context
        conversation_history = self._format_conversation_history()
        if conversation_history:
            prompt = (
                f"{conversation_history}\n\n"
                f"Based on the conversation above, answer this question directly and concisely:\n\n"
                f"{task}\n\nAnswer:"
            )
        else:
            prompt = f"Answer this question directly and concisely:\n\n{task}\n\nAnswer:"

        try:
            response = self._generate(prompt, **kwargs)
            answer = response["text"].strip()
            tokens_used = response.get("tokens_used", 0)

            return AgentResponse(
                output=answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            return AgentResponse(
                output=f"Failed to answer: {str(e)}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"error": str(e)}
            )

    def _get_tools_description(self, verbose: bool | None = None) -> str:
        """
        Get formatted description of available tools.

        Args:
            verbose: Override verbosity. If None, uses self._verbose_tools.

        Returns:
            Formatted tools description string.
        """
        if not self.tools:
            return "No tools available."

        use_verbose = verbose if verbose is not None else self._verbose_tools
        return self._tool_prompt_generator.generate_tools_section(verbose=use_verbose)

    def _format_conversation_history(self, max_turns: int = 25) -> str:
        """
        Format conversation history for inclusion in prompt.

        Uses ShortTermMemory to retrieve recent messages, including
        summaries of older messages when available.

        Args:
            max_turns: Maximum number of previous turns (user+assistant pairs)

        Returns:
            Formatted conversation history string
        """
        # Include summaries of older messages first
        summaries = self.short_term_memory.summaries
        messages = self.short_term_memory.get_recent_messages(n=max_turns * 2)
        if not messages and not summaries:
            return ""

        history = "\n\n=== Previous Conversation Context ===\n"

        # Add summaries if they exist (these cover older, summarized turns)
        if summaries:
            for summary in summaries:
                history += f"[Earlier context summary: {summary.summary}]\n\n"

        turn_num = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == MessageRole.USER:
                turn_num += 1
                history += f"[Turn {turn_num}]\n"
                history += f"User: {msg.content}\n"
                # Check if next message is assistant
                if i + 1 < len(messages) and messages[i + 1].role == MessageRole.ASSISTANT:
                    # Truncate long assistant responses to save tokens
                    assistant_content = messages[i + 1].content
                    if len(assistant_content) > 300:
                        assistant_content = assistant_content[:300] + "..."
                    history += f"Assistant: {assistant_content}\n\n"
                    i += 2
                    continue
                else:
                    history += "\n"
            i += 1

        history += "=== End of Previous Context ===\n"
        return history if turn_num > 0 or summaries else ""

    def add_tool(self, tool: BaseTool):
        """
        Add a tool to the agent.

        Args:
            tool: Tool instance to add
        """
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]

    def reset_memory(self):
        """Clear conversation and tool history."""
        self.state.clear_history()
        self.short_term_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.end_session()
            self.long_term_memory.start_session(name=self.name)

    def save_state(self, filepath: str, format: str = "json"):
        """
        Save agent state.

        Args:
            filepath: Path to save to
            format: Format (json or pickle)
        """
        self.state.save(filepath, format)

    def load_state(self, filepath: str, format: str = "json"):
        """
        Load agent state.

        Args:
            filepath: Path to load from
            format: Format (json or pickle)
        """
        self.state = AgentState.load(filepath, format)

    def synthesize(self, synthesis_data: dict[str, Any]) -> str:
        """
        Synthesize results from sub-agents.

        Args:
            synthesis_data: Data to synthesize

        Returns:
            Synthesized output
        """
        # Build synthesis prompt
        results_text = []
        for result in synthesis_data.get("results", []):
            output = result.get("output", {})
            if isinstance(output, dict):
                results_text.append(output.get("output", str(output)))
            else:
                results_text.append(str(output))

        prompt = f"""Synthesize the following results into a comprehensive answer for: {synthesis_data['original_task']}

Results from sub-agents:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(results_text))}

Provide a well-structured, comprehensive response that integrates all findings."""

        # Generate synthesis
        response = self._generate(prompt, temperature=0.6)
        return response.get("text", "").strip()

    async def run_async(self,
                       task: str,
                       mode: AgentMode = AgentMode.AUTO,
                       context: dict[str, Any] | None = None,
                       **kwargs) -> AgentResponse:
        """
        Truly asynchronous version of run().

        Runs the synchronous run() method in a thread executor so it
        doesn't block the event loop, while remaining compatible with
        async callers.

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        import functools
        loop = asyncio.get_running_loop()
        func = functools.partial(self.run, task, mode, context, **kwargs)
        return await loop.run_in_executor(None, func)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit — clean up resources."""
        # Reset circuit breaker
        self._circuit_breaker.reset_all()
        # Clear short-term memory if desired
        return False

    def stream(self,
               task: str,
               mode: AgentMode = AgentMode.AUTO,
               context: dict[str, Any] | None = None,
               on_thought: Callable[[str], None] | None = None,
               on_tool_call: Callable[[str, str], None] | None = None,
               on_observation: Callable[[str], None] | None = None,
               on_answer: Callable[[str], None] | None = None,
               **kwargs) -> Iterator[str]:
        """
        Stream response token by token using real model streaming.

        Streams the ReAct loop in real-time:
        - Yields thought tokens as they generate
        - Pauses streaming during tool execution
        - Yields observation text after tool calls
        - Yields final answer tokens

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            on_thought: Callback for thought tokens
            on_tool_call: Callback(tool_name, tool_input) when a tool is called
            on_observation: Callback for tool observation text
            on_answer: Callback for final answer tokens
            **kwargs: Additional arguments

        Yields:
            Response tokens (str)
        """
        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        context = context or {}
        max_iterations = self.config.max_iterations
        scratchpad = ""
        iterations = 0
        tool_calls = 0

        # Build conversation history
        conversation_history = self._format_conversation_history()

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
            "\n\n\n",
        ]

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences", default_stop_sequences),
        )

        while iterations < max_iterations:
            iterations += 1

            # Build prompt
            tools_desc = self._get_tools_description()
            if self.config.system_prompt_template:
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_desc,
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                )
            else:
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                )

            # Stream tokens from model
            accumulated = ""
            try:
                for token in self.model.generate_stream(prompt, config=gen_config):
                    accumulated += token

                    # Check for stop sequences
                    hit_stop = False
                    for stop_seq in default_stop_sequences:
                        if stop_seq in accumulated:
                            # Trim at stop sequence
                            idx = accumulated.index(stop_seq)
                            accumulated = accumulated[:idx]
                            hit_stop = True
                            break

                    yield token

                    if hit_stop:
                        break

            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                yield f"\n[Error: {e}]"
                return

            # Parse the accumulated response
            parsed = self._parse_react_response(accumulated)
            thought = parsed.get("thought", "")
            scratchpad += f"\nThought: {thought}"

            if on_thought and thought:
                on_thought(thought)

            # Check for final answer
            if parsed.get("final_answer"):
                answer = parsed["final_answer"]
                if on_answer:
                    on_answer(answer)
                # Store in memory
                if answer:
                    self.short_term_memory.add_user_message(task)
                    self.short_term_memory.add_assistant_message(answer)
                return

            # Execute tool if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                if on_tool_call:
                    on_tool_call(action, action_input)

                if action in self.tools:
                    tool_result = self._execute_tool(action, action_input)
                    tool_calls += 1

                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Yield observation
                    obs_text = f"\nObservation: {tool_result}\n"
                    yield obs_text
                    if on_observation:
                        on_observation(str(tool_result))
                else:
                    no_tool_msg = f"\nObservation: Tool '{action}' not found. Use 'Final Answer:' to respond directly.\n"
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: Tool '{action}' not found."
                    yield no_tool_msg
            else:
                scratchpad += "\nAction: (continue reasoning)"

        yield "\n[Max iterations reached]"

    def get_execution_summary(self) -> dict[str, Any]:
        """
        Get summary of execution.

        Returns:
            Summary dictionary
        """
        return self.execution_tracker.get_summary()

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name={self.name}, tools={len(self.tools)}, sub_agents={self.config.enable_sub_agents})"
