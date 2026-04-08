"""
Typed Pydantic models for the Data Cleaning Environment.

Defines Action, Observation, and State models used by the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# =============================================================================
# Action Model
# =============================================================================

class DataCleaningAction(Action):
    """An action the agent can take to clean the dataset.

    Supported action types:
        - "remove_duplicates": Remove duplicate rows (optionally by subset of columns)
        - "fill_missing": Fill missing values in a column
        - "standardize_format": Standardize string formatting in a column
        - "fix_outliers": Cap or remove statistical outliers in a numeric column
        - "rename_column": Rename a column header
        - "drop_column": Drop an irrelevant column
        - "correct_typos": Fix common typos/inconsistencies in categorical column
        - "convert_type": Convert column data type
        - "submit": Submit the cleaned dataset for grading
    """

    action_type: str = Field(
        ...,
        description="Type of cleaning operation to perform",
    )
    column: Optional[str] = Field(
        default=None,
        description="Target column name for the operation",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the action (e.g., fill_value, strategy)",
    )


# =============================================================================
# Observation Model
# =============================================================================

class DataCleaningObservation(Observation):
    """Observation returned after each step in the data cleaning environment.

    Provides the agent with information about the current state of the dataset
    and the result of the last action.
    """

    # Dataset summary
    num_rows: int = Field(default=0, description="Number of rows in the dataset")
    num_columns: int = Field(default=0, description="Number of columns in the dataset")
    column_names: List[str] = Field(default_factory=list, description="List of column names")
    column_types: Dict[str, str] = Field(default_factory=dict, description="Column name -> data type mapping")

    # Data quality metrics
    missing_value_counts: Dict[str, int] = Field(
        default_factory=dict, description="Number of missing values per column"
    )
    duplicate_row_count: int = Field(default=0, description="Number of duplicate rows")
    
    # Sample data (first 5 rows as list of dicts)
    sample_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="First 5 rows of the dataset"
    )

    # Column statistics for numeric columns
    column_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistics (mean, std, min, max) for numeric columns",
    )

    # Data quality issues detected
    detected_issues: List[str] = Field(
        default_factory=list,
        description="List of detected data quality issues",
    )

    # Action feedback
    last_action_success: bool = Field(
        default=True, description="Whether the last action succeeded"
    )
    last_action_message: str = Field(
        default="", description="Feedback message from the last action"
    )

    # Task info
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Description of the cleaning task")
    max_steps: int = Field(default=20, description="Maximum steps allowed")
    current_step: int = Field(default=0, description="Current step number")


# =============================================================================
# State Model
# =============================================================================

class DataCleaningState(State):
    """Internal state of the data cleaning environment."""

    task_id: str = Field(default="", description="Current task identifier")
    current_step: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=20, description="Maximum steps per episode")
    is_done: bool = Field(default=False, description="Whether the episode is finished")
    current_score: float = Field(default=0.0, description="Current quality score (0.0-1.0)")
    actions_taken: List[str] = Field(default_factory=list, description="History of actions taken")
