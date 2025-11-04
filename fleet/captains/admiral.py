"""
Admiral - Plan-and-Execute Pattern

An Admiral uses the Plan-and-Execute pattern: creating a comprehensive
plan upfront, then systematically executing each step.

Plan → Execute Step 1 → Execute Step 2 → ... → Complete
"""

from typing import Optional, List, Dict, Any
from fleet.captains.tool_captain import ToolCaptain
from fleet.providers.base_provider import BaseProvider
from fleet.tools.toolbox import Toolbox
from fleet.tools.tool import Tool
from fleet.payload.payload import Payload
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStep:
    """Represents one step in the execution plan"""
    step_number: int
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


class Admiral(ToolCaptain):
    """
    Admiral implements the Plan-and-Execute pattern.

    The Admiral creates a comprehensive plan before execution,
    then systematically executes each step with optional replanning.

    Perfect for:
    - Complex multi-step tasks
    - When plan can be determined upfront
    - Cost optimization (plan once, execute with cheaper model)
    - Tasks requiring approval before execution
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "Admiral",
        system_prompt: str = None,
        description: str = "",
        color: str = "blue",
        toolbox: Optional[Toolbox] = None,
        tools: Optional[List[Tool]] = None,
        planner_model: Optional[str] = None,
        executor_model: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        allow_replanning: bool = True,
        verbose: bool = True
    ):
        """
        Initialize an Admiral (Plan-and-Execute agent).

        Args:
            provider: The LLM provider
            name: Admiral name
            system_prompt: System instructions (auto-generated if None)
            description: Admiral's role
            color: Terminal color
            toolbox: Tools available
            tools: Individual instruments
            planner_model: Model for planning (can be more powerful)
            executor_model: Model for execution (can be cheaper)
            default_temperature: Default temperature
            default_max_tokens: Default max tokens
            allow_replanning: Whether to allow plan updates during execution
            verbose: Whether to print execution details
        """
        if system_prompt is None:
            system_prompt = self._create_planner_prompt()

        super().__init__(
            provider=provider,
            name=name,
            system_prompt=system_prompt,
            description=description,
            color=color,
            toolbox=toolbox,
            tools=tools,
            default_model=planner_model or executor_model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens
        )

        self.planner_model = planner_model
        self.executor_model = executor_model or planner_model
        self.allow_replanning = allow_replanning
        self.verbose = verbose

        logger.info(f"Initialized Admiral: {name} with Plan-and-Execute pattern")

    def _create_planner_prompt(self) -> str:
        """Create a planning-focused system prompt"""
        return """You are an Admiral who excels at strategic planning.

When given a task:
1. Break it down into clear, actionable steps
2. Consider dependencies between steps
3. Identify tools/resources needed for each step
4. Create a logical execution order

Be thorough and systematic in your planning."""

    def command(
        self,
        mission: str,
        planner_model: Optional[str] = None,
        executor_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Payload:
        """
        Command the execution of a mission using Plan-and-Execute pattern.

        Args:
            mission: The mission/task to accomplish
            planner_model: Model for planning phase
            executor_model: Model for execution phase
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options

        Returns:
            Payload with results and execution trace
        """
        planner_model = planner_model or self.planner_model or self.default_model
        executor_model = executor_model or self.executor_model or self.default_model

        if planner_model is None:
            raise ValueError("No planner model specified")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⚓ Admiral: {self.name}")
            print(f"Mission: {mission}")
            print(f"{'='*60}\n")

        # Phase 1: Planning
        if self.verbose:
            print(f"📋 Phase 1: Strategic Planning")
            print(f"Planner Model: {planner_model}\n")

        plan = self._create_plan(mission, planner_model, temperature, max_tokens, **kwargs)
        steps = self._parse_plan(plan.content)

        if self.verbose:
            print(f"✓ Plan Created: {len(steps)} steps\n")
            for step in steps:
                print(f"  {step.step_number}. {step.description}")
            print()

        # Phase 2: Execution
        if self.verbose:
            print(f"⚡ Phase 2: Execution")
            print(f"Executor Model: {executor_model}\n")

        execution_results = []
        for step in steps:
            step.status = "in_progress"
            if self.verbose:
                print(f"🎯 Step {step.step_number}: {step.description}")

            try:
                result = self._execute_step(
                    step,
                    mission,
                    steps,
                    executor_model,
                    temperature,
                    max_tokens,
                    **kwargs
                )

                step.status = "completed"
                step.result = result.content
                step.input_tokens = result.input_tokens
                step.output_tokens = result.output_tokens

                execution_results.append(result)

                if self.verbose:
                    print(f"✓ Completed: {result.content[:80]}...\n")

            except Exception as e:
                step.status = "failed"
                step.result = f"Error: {str(e)}"
                logger.error(f"Step {step.step_number} failed: {e}")

                if self.allow_replanning:
                    if self.verbose:
                        print(f"⚠️  Step failed, attempting to replan...\n")
                    # In a full implementation, you'd replan here
                    pass
                else:
                    raise

        # Create final payload
        total_input_tokens = plan.input_tokens + sum(r.input_tokens for r in execution_results)
        total_output_tokens = plan.output_tokens + sum(r.output_tokens for r in execution_results)

        # Synthesize final result
        final_content = self._synthesize_results(steps, mission)

        final_payload = Payload(
            content=final_content,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            metadata={
                "pattern": "Plan-and-Execute",
                "plan": plan.content,
                "steps": [
                    {
                        "step": s.step_number,
                        "description": s.description,
                        "status": s.status,
                        "result": s.result
                    }
                    for s in steps
                ],
                "planner_model": planner_model,
                "executor_model": executor_model
            },
            captain_name=self.name
        )

        if self.verbose:
            print(f"{'='*60}")
            print(f"✓ Mission Complete!")
            print(f"Total Steps: {len(steps)}")
            print(f"Successful: {sum(1 for s in steps if s.status == 'completed')}")
            print(f"Total Tokens: {final_payload.total_tokens}")
            print(f"{'='*60}\n")

        return final_payload

    def _create_plan(
        self,
        mission: str,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs
    ) -> Payload:
        """Create the execution plan"""
        planning_prompt = f"""Create a detailed step-by-step plan to accomplish this mission:

Mission: {mission}

Provide a numbered list of steps. Be specific and actionable.
Consider what tools might be needed and any dependencies between steps.

Format each step as:
1. [Clear description of what to do]
2. [Next step]
etc."""

        self.clear_messages()
        return self.send_message(
            content=planning_prompt,
            model=model,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            tool_choice="none",  # No tools during planning
            **kwargs
        )

    def _parse_plan(self, plan_text: str) -> List[ExecutionStep]:
        """Parse the plan text into structured steps"""
        lines = plan_text.strip().split('\n')
        steps = []
        step_number = 0

        for line in lines:
            line = line.strip()
            # Look for numbered steps (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                step_number += 1
                # Clean up the line
                description = line.lstrip('0123456789.-•) ').strip()
                if description:
                    steps.append(ExecutionStep(
                        step_number=step_number,
                        description=description,
                        status="pending"
                    ))

        return steps

    def _execute_step(
        self,
        step: ExecutionStep,
        mission: str,
        all_steps: List[ExecutionStep],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs
    ) -> Payload:
        """Execute a single step"""
        # Build context with mission and previous results
        context = f"Mission: {mission}\n\n"
        context += "Previous steps completed:\n"
        for s in all_steps:
            if s.step_number < step.step_number and s.status == "completed":
                context += f"{s.step_number}. {s.description}\n   Result: {s.result}\n"

        execution_prompt = f"""{context}

Now execute this step:
{step.step_number}. {step.description}

Provide the result of executing this step."""

        self.clear_messages()
        return self.send_message(
            content=execution_prompt,
            model=model,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            **kwargs
        )

    def _synthesize_results(self, steps: List[ExecutionStep], mission: str) -> str:
        """Synthesize all step results into final output"""
        result_parts = [f"Mission: {mission}\n"]
        result_parts.append(f"Completed {len(steps)} steps:\n")

        for step in steps:
            result_parts.append(f"\n{step.step_number}. {step.description}")
            if step.result:
                result_parts.append(f"   → {step.result}")

        return "\n".join(result_parts)

    def __str__(self) -> str:
        return f"Admiral({self.name}, Plan-and-Execute pattern)"
