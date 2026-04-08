"""
FastAPI server for the Data Cleaning Environment.

Exposes the OpenEnv-compliant HTTP API: /reset, /step, /state, /health
Includes stateful session management for HTTP-based inference.
"""

import os
import sys
import uuid

# Add parent directory to path so models.py can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.types import HealthResponse, HealthStatus

from environment import DataCleaningEnvironment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DataCleaningAction, DataCleaningObservation

# Create FastAPI app
app = FastAPI(
    title="Data Cleaning Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to clean "
        "messy tabular datasets. Supports 3 tasks: basic cleaning (easy), "
        "format standardization (medium), and complex multi-issue cleaning (hard)."
    ),
    version="1.0.0",
)

# =============================================================================
# Stateful session management for HTTP endpoints  
# The default HTTPEnvServer HTTP endpoints are stateless (new env per request).
# We add a simple in-memory session to support stateful /reset, /step, /state.
# =============================================================================

# Single shared environment instance for stateful HTTP interaction
_env_instance = DataCleaningEnvironment()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = "task_easy"
    
    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None
    
    class Config:
        extra = "allow"


def _serialize_observation(obs: DataCleaningObservation) -> dict:
    """Serialize observation to JSON-compatible dict."""
    obs_dict = obs.model_dump()
    return {
        "observation": obs_dict,
        "reward": obs_dict.get("reward"),
        "done": obs_dict.get("done", False),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)):
    """Reset the environment with a specific task."""
    global _env_instance
    _env_instance = DataCleaningEnvironment()
    
    obs = _env_instance.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    return _serialize_observation(obs)


@app.post("/step")
async def step(request: StepRequest):
    """Execute an action in the environment."""
    global _env_instance
    
    action_data = request.action
    action = DataCleaningAction(**action_data)
    
    obs = _env_instance.step(action, timeout_s=request.timeout_s)
    return _serialize_observation(obs)


@app.get("/state")
async def state():
    """Get the current environment state."""
    global _env_instance
    state_data = _env_instance.state
    return state_data.model_dump()


@app.get("/schema")
async def schema():
    """Get the action/observation/state schemas."""
    return {
        "action": DataCleaningAction.model_json_schema(),
        "observation": DataCleaningObservation.model_json_schema(),
        "state": DataCleaningObservation.model_json_schema(),
    }


@app.get("/metadata")
async def metadata():
    """Get environment metadata."""
    return {
        "name": "DataCleaningEnvironment",
        "description": "An environment for training AI agents on real-world data cleaning tasks",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
    }


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
