"""
Base Pipeline Stage
Abstract base class for all training pipeline stages.
Part of Phase 2.2: Code Organization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """
    Result from a pipeline stage execution.

    Attributes:
        stage_name: Name of the stage
        success: Whether stage completed successfully
        duration_seconds: Time taken to execute
        data: Output data from the stage
        metrics: Performance metrics
        errors: List of errors encountered
        warnings: List of warnings
    """
    stage_name: str
    success: bool
    duration_seconds: float
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"[{self.stage_name}] {status} ({self.duration_seconds:.2f}s)"


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage is responsible for:
    - One specific task in the training pipeline
    - Validation of inputs
    - Error handling
    - Progress reporting
    - Metrics collection
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> StageResult:
        """
        Execute the stage.

        Args:
            context: Execution context containing input data and state

        Returns:
            StageResult with output data and metrics

        The context dictionary typically contains:
        - 'audio_file': Path to input audio file
        - 'config': Configuration object
        - 'previous_results': Results from previous stages
        - Any stage-specific inputs
        """
        pass

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        """
        Validate inputs before execution.

        Args:
            context: Execution context

        Returns:
            List of validation errors (empty if valid)
        """
        return []

    def run(self, context: Dict[str, Any]) -> StageResult:
        """
        Run the stage with timing and error handling.

        Args:
            context: Execution context

        Returns:
            StageResult
        """
        self.logger.info(f"Starting stage: {self.name}")
        start_time = time.time()

        try:
            # Validate inputs
            errors = self.validate_inputs(context)
            if errors:
                duration = time.time() - start_time
                self.logger.error(f"Validation failed: {errors}")
                return StageResult(
                    stage_name=self.name,
                    success=False,
                    duration_seconds=duration,
                    errors=errors
                )

            # Execute stage
            result = self.execute(context)

            # Add timing if not already set
            if result.duration_seconds == 0:
                result.duration_seconds = time.time() - start_time

            self.logger.info(f"Completed stage: {self.name} ({result.duration_seconds:.2f}s)")
            return result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Stage failed: {self.name}")

            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=duration,
                errors=[f"Exception: {str(e)}"]
            )

    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input keys from context.

        Returns:
            List of required context keys
        """
        return []

    def get_output_keys(self) -> List[str]:
        """
        Get list of output keys this stage produces.

        Returns:
            List of output keys in result.data
        """
        return []
