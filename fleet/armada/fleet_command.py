"""
Fleet Command - Hierarchical Pattern

Fleet Command implements hierarchical agent orchestration where a
director/manager coordinates multiple specialized worker agents.

Director → Worker 1, Worker 2, Worker 3 → Synthesis
"""

from typing import List, Union, Optional, Dict, Any
from fleet.captains.base_captain import BaseCaptain
from fleet.payload.payload import Payload
from termcolor import colored
import logging

logger = logging.getLogger(__name__)


class FleetCommand:
    """
    Fleet Command implements the Hierarchical (Director-Worker) pattern.

    A director captain coordinates multiple specialized worker captains,
    decomposing tasks and synthesizing results.

    Perfect for:
    - Large complex projects
    - Task decomposition
    - Specialized agent expertise
    - Clear hierarchical structure
    """

    def __init__(
        self,
        director: BaseCaptain,
        workers: List[BaseCaptain],
        name: str = "Fleet Command",
        description: str = "",
        task_decomposition_strategy: str = "llm",
        synthesize_results: bool = True
    ):
        """
        Initialize Fleet Command (Hierarchical orchestration).

        Args:
            director: The director/manager captain
            workers: List of worker captains
            name: Fleet Command name
            description: Description
            task_decomposition_strategy: "llm" (director decides) or "rules"
            synthesize_results: Whether director synthesizes final results
        """
        self.director = director
        self.workers = workers
        self.name = name
        self.description = description
        self.task_decomposition_strategy = task_decomposition_strategy
        self.synthesize_results = synthesize_results

        # Assign colors to workers
        colors = ['cyan', 'yellow', 'green', 'magenta', 'blue', 'red']
        for i, worker in enumerate(workers):
            if not hasattr(worker, 'color') or worker.color == 'white':
                worker.color = colors[i % len(colors)]

        logger.info(f"Initialized Fleet Command: {name} with director + {len(workers)} workers")

    def execute_mission(
        self,
        mission: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Payload:
        """
        Execute a mission using hierarchical orchestration.

        Args:
            mission: The mission to accomplish
            model: Model to use for all captains
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options

        Returns:
            Payload with synthesized results
        """
        print(f"\n{'='*60}")
        print(colored(f"⚓ FLEET COMMAND: {self.name}", 'white', attrs=['bold']))
        print(f"Mission: {mission}")
        print(f"{'='*60}\n")

        # Phase 1: Director decomposes the task
        print(colored("📋 Phase 1: Director Planning", 'white', attrs=['bold']))
        print(f"Director: {self.director.name}\n")

        task_assignments = self._decompose_task(mission, model, temperature, max_tokens, **kwargs)

        print(f"✓ Task Decomposition Complete")
        print(f"Assignments: {len(task_assignments)}\n")

        # Phase 2: Workers execute their tasks
        print(colored("⚡ Phase 2: Worker Execution", 'white', attrs=['bold']))

        worker_results = []
        for assignment in task_assignments:
            worker = assignment['worker']
            task = assignment['task']

            print(colored(f"\n🎯 {worker.name}: {task[:60]}...", worker.color))

            try:
                if hasattr(worker, 'chat'):
                    result = worker.chat(task, model=model)
                else:
                    result = worker.send_message(task, model, temperature, max_tokens, **kwargs)

                worker_results.append({
                    'worker': worker.name,
                    'task': task,
                    'result': result,
                    'success': True
                })

                print(colored(f"✓ {worker.name} Complete: {result.content[:60]}...", worker.color))

            except Exception as e:
                logger.error(f"Worker {worker.name} failed: {e}")
                worker_results.append({
                    'worker': worker.name,
                    'task': task,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
                print(colored(f"✗ {worker.name} Failed: {str(e)}", 'red'))

        # Phase 3: Director synthesizes results
        print(f"\n{colored('📊 Phase 3: Director Synthesis', 'white', attrs=['bold'])}\n")

        if self.synthesize_results:
            final_result = self._synthesize_results(
                mission,
                worker_results,
                model,
                temperature,
                max_tokens,
                **kwargs
            )

            print(colored(f"✓ Synthesis Complete", 'white', attrs=['bold']))
        else:
            # No synthesis, just aggregate
            final_content = self._aggregate_results(mission, worker_results)
            total_input_tokens = sum(
                wr['result'].input_tokens for wr in worker_results if wr['success']
            )
            total_output_tokens = sum(
                wr['result'].output_tokens for wr in worker_results if wr['success']
            )

            final_result = Payload(
                content=final_content,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                metadata={
                    "pattern": "Hierarchical",
                    "worker_results": worker_results
                }
            )

        print(f"\n{'='*60}")
        print(colored(f"✓ MISSION COMPLETE", 'green', attrs=['bold']))
        print(f"Total Tokens: {final_result.total_tokens}")
        print(f"{'='*60}\n")

        return final_result

    def _decompose_task(
        self,
        mission: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Director decomposes the mission into worker tasks"""

        worker_descriptions = "\n".join([
            f"- {worker.name}: {worker.description or 'General agent'}"
            for worker in self.workers
        ])

        decomposition_prompt = f"""You are a director managing a team. Decompose this mission into specific tasks for your team members.

Mission: {mission}

Available team members:
{worker_descriptions}

For each team member that should be involved, provide a specific task.
Format your response as a list:

1. [Worker Name]: [Specific task]
2. [Worker Name]: [Specific task]
etc.

Only assign tasks to workers that are relevant. Not all workers need to be involved."""

        self.director.clear_messages()
        if hasattr(self.director, 'chat'):
            decomp_result = self.director.chat(decomposition_prompt, model=model)
        else:
            decomp_result = self.director.send_message(
                decomposition_prompt,
                model,
                temperature,
                max_tokens,
                **kwargs
            )

        # Parse the decomposition
        assignments = self._parse_decomposition(decomp_result.content)

        return assignments

    def _parse_decomposition(self, decomp_text: str) -> List[Dict[str, Any]]:
        """Parse director's decomposition into structured assignments"""
        lines = decomp_text.strip().split('\n')
        assignments = []

        for line in lines:
            line = line.strip()
            if not line or not any(char.isdigit() for char in line[:3]):
                continue

            # Try to find worker name and task
            # Format: "1. Worker Name: Task description"
            if ':' in line:
                parts = line.split(':', 1)
                worker_part = parts[0].strip().lstrip('0123456789.-•) ')
                task_part = parts[1].strip()

                # Find matching worker
                worker = self._find_worker_by_name(worker_part)
                if worker:
                    assignments.append({
                        'worker': worker,
                        'task': task_part
                    })

        # If parsing failed, distribute work evenly
        if not assignments and self.workers:
            logger.warning("Failed to parse decomposition, distributing work evenly")
            for worker in self.workers:
                assignments.append({
                    'worker': worker,
                    'task': f"Help with: {decomp_text[:100]}"
                })

        return assignments

    def _find_worker_by_name(self, name_part: str) -> Optional[BaseCaptain]:
        """Find a worker whose name matches the given string"""
        name_part_lower = name_part.lower()
        for worker in self.workers:
            if worker.name.lower() in name_part_lower or name_part_lower in worker.name.lower():
                return worker
        return None

    def _synthesize_results(
        self,
        mission: str,
        worker_results: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Payload:
        """Director synthesizes worker results"""

        results_text = "\n\n".join([
            f"Worker: {wr['worker']}\nTask: {wr['task']}\nResult: {wr['result'].content if wr['success'] else 'Failed'}"
            for wr in worker_results
        ])

        synthesis_prompt = f"""You are a director synthesizing your team's work.

Original Mission: {mission}

Team Results:
{results_text}

Synthesize these results into a comprehensive, cohesive response to the mission.
Integrate all successful worker contributions into a clear final answer."""

        self.director.clear_messages()
        if hasattr(self.director, 'chat'):
            synthesis_result = self.director.chat(synthesis_prompt, model=model)
        else:
            synthesis_result = self.director.send_message(
                synthesis_prompt,
                model,
                temperature,
                max_tokens,
                **kwargs
            )

        # Add metadata
        synthesis_result.metadata["pattern"] = "Hierarchical"
        synthesis_result.metadata["worker_count"] = len(worker_results)
        synthesis_result.metadata["successful_workers"] = sum(1 for wr in worker_results if wr['success'])

        return synthesis_result

    def _aggregate_results(self, mission: str, worker_results: List[Dict]) -> str:
        """Simple aggregation without synthesis"""
        parts = [f"Mission: {mission}\n"]
        for wr in worker_results:
            parts.append(f"\n{wr['worker']}:")
            if wr['success']:
                parts.append(f"  {wr['result'].content}")
            else:
                parts.append(f"  Failed: {wr.get('error', 'Unknown error')}")

        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.workers) + 1  # workers + director

    def __str__(self) -> str:
        return f"FleetCommand({self.name}, Director + {len(self.workers)} workers)"
