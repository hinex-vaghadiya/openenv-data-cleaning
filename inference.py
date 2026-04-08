"""
Baseline Inference Script for the Data Cleaning Environment.

Uses the OpenAI API client (compatible with any OpenAI-API-compatible endpoint)
to run an LLM agent against all 3 tasks, producing reproducible baseline scores.

Required environment variables:
  - API_BASE_URL:   The API endpoint (e.g., https://api.openai.com/v1)
  - MODEL_NAME:     The model identifier (e.g., gpt-4o-mini)
  - HF_TOKEN:       Your API key (used as OPENAI_API_KEY)
  - OPENAI_API_KEY: Alternative API key variable

Stdout format follows the mandatory [START], [STEP], [END] structure.
"""

import json
import os
import re
import sys
import time

import requests
from openai import OpenAI

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy-key-for-init")
ENV_URL = os.environ.get("ENV_URL")
if not ENV_URL:
    # Try to auto-discover if running on Hugging Face or known space
    space_id = os.environ.get("SPACE_ID")
    if space_id:
        # e.g. "username/space-name" -> "https://username-space-name.hf.space"
        host = space_id.replace("/", "-").lower()
        ENV_URL = f"https://{host}.hf.space"
    else:
        # Final fallback
        ENV_URL = "https://hinex-07-data-cleaning-env.hf.space"

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# =============================================================================
# Environment interaction helpers
# =============================================================================

def env_reset(task_id: str = "task_easy", seed: int = 42) -> dict:
    """Reset the environment for a specific task."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"seed": seed, "task_id": task_id},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Execute an action in the environment."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    """Get the current environment state."""
    resp = requests.get(f"{ENV_URL}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# LLM Agent
# =============================================================================

SYSTEM_PROMPT = """You are an expert data cleaning agent. You are given a messy dataset and must clean it by performing a series of actions.

Available action types:
1. "remove_duplicates" - Remove duplicate rows. Optionally specify a "column" for subset-based dedup.
2. "fill_missing" - Fill missing values. Requires "column". Params: "strategy" (drop/fill_value/mean/mode), "fill_value" (if strategy=fill_value)
3. "standardize_format" - Standardize column format. Requires "column". Params: "format" (iso_date/phone_standard/lowercase/uppercase/titlecase/strip_whitespace)
4. "fix_outliers" - Fix statistical outliers. Requires "column". Params: "strategy" (clip/remove), "lower_bound", "upper_bound"
5. "rename_column" - Rename a column. Requires "column". Params: "new_name"
6. "drop_column" - Drop an irrelevant column. Requires "column".
7. "correct_typos" - Fix typos in categorical column. Requires "column". Params: "mapping" (dict of old_value -> new_value)
8. "convert_type" - Convert column data type. Requires "column". Params: "target_type" (float/int/str), "strip_chars" (chars to remove before conversion, e.g. "$,")
9. "submit" - Submit the cleaned dataset for final grading. Use when you're confident the data is clean.

You MUST respond with a valid JSON action object. Example:
{"action_type": "remove_duplicates", "column": null, "params": {}}
{"action_type": "fill_missing", "column": "email", "params": {"strategy": "drop"}}
{"action_type": "correct_typos", "column": "status", "params": {"mapping": {"Compelted": "completed", "COMPLETED": "completed"}}}
{"action_type": "submit", "column": null, "params": {}}

Respond ONLY with the JSON object, no other text. Analyze the observation carefully and choose the most impactful action.
"""


def parse_action(response_text: str) -> dict:
    """Extract a JSON action from the LLM response."""
    # Try to find JSON in the response
    text = response_text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object pattern
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fall back to submit
    return {"action_type": "submit", "column": None, "params": {}}


def format_observation_for_llm(obs: dict) -> str:
    """Format the observation into a human-readable prompt for the LLM."""
    o = obs.get("observation", obs)
    
    lines = []
    lines.append(f"=== Dataset Overview ===")
    lines.append(f"Task: {o.get('task_id', 'unknown')} - {o.get('task_description', '')}")
    lines.append(f"Step: {o.get('current_step', 0)} / {o.get('max_steps', 20)}")
    lines.append(f"Rows: {o.get('num_rows', 0)}, Columns: {o.get('num_columns', 0)}")
    lines.append(f"Duplicate rows: {o.get('duplicate_row_count', 0)}")
    lines.append(f"Quality Score: {o.get('metadata', {}).get('quality_score', 'N/A')}")
    lines.append(f"Last action: {'[OK]' if o.get('last_action_success', True) else '[FAIL]'} {o.get('last_action_message', '')}")
    
    lines.append(f"\n=== Column Info ===")
    lines.append(f"Columns: {o.get('column_names', [])}")
    lines.append(f"Types: {json.dumps(o.get('column_types', {}), indent=2)}")
    
    missing = o.get('missing_value_counts', {})
    if missing:
        lines.append(f"\n=== Missing Values ===")
        for col, count in missing.items():
            lines.append(f"  {col}: {count} missing")
    
    stats = o.get('column_stats', {})
    if stats:
        lines.append(f"\n=== Numeric Column Stats ===")
        for col, s in stats.items():
            lines.append(f"  {col}: mean={s.get('mean')}, std={s.get('std')}, min={s.get('min')}, max={s.get('max')}")
    
    issues = o.get('detected_issues', [])
    if issues:
        lines.append(f"\n=== Detected Issues ===")
        for issue in issues:
            lines.append(f"  [!] {issue}")
    
    sample = o.get('sample_data', [])
    if sample:
        lines.append(f"\n=== Sample Data (first {len(sample)} rows) ===")
        for i, row in enumerate(sample):
            lines.append(f"  Row {i}: {json.dumps(row, default=str)}")
    
    return "\n".join(lines)


def run_task(task_id: str, seed: int = 42, max_retries: int = 3) -> dict:
    """Run the agent on a single task and return results."""
    
    # [START] log
    start_time = time.time()
    task_start_json = {
        "type": "[START]",
        "task_id": task_id,
        "seed": seed,
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(task_start_json))
    sys.stdout.flush()
    
    # Reset environment
    try:
        obs = env_reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"  Failed to reset environment: {e}", file=sys.stderr)
        # Emit [END] immediately on reset failure
        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "score": 0.0,
            "total_reward": 0.0,
            "steps": 0,
            "error": f"Reset failed: {str(e)}",
            "model": MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        sys.stdout.flush()
        return {"type": "[END]", "task_id": task_id, "score": 0.0}
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    total_reward = 0.0
    step_count = 0
    done = False
    final_score = 0.0
    
    while not done:
        # Format observation for LLM
        obs_text = format_observation_for_llm(obs)
        messages.append({"role": "user", "content": obs_text})
        
        # Get LLM response
        action_dict = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
                response_text = response.choices[0].message.content.strip()
                action_dict = parse_action(response_text)
                messages.append({"role": "assistant", "content": response_text})
                break
            except Exception as e:
                print(f"  LLM call failed (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr)
                if attempt == max_retries - 1:
                    # Fall back to submit
                    action_dict = {"action_type": "submit", "column": None, "params": {}}
        
        # Execute action
        step_count += 1
        obs = env_step(action_dict)
        
        obs_data = obs.get("observation", obs)
        reward = obs.get("reward", 0.0)
        done = obs.get("done", False)
        total_reward += reward if reward else 0.0
        quality_score = obs_data.get("metadata", {}).get("quality_score", 0.0)
        
        # [STEP] log
        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": step_count,
            "action": action_dict.get("action_type", "unknown"),
            "reward": reward,
            "quality_score": quality_score,
            "done": done,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        sys.stdout.flush()
        
        if done:
            final_score = quality_score
            break
        
        # Keep conversation history manageable
        if len(messages) > 20:
            messages = messages[:2] + messages[-10:]
    
    elapsed = time.time() - start_time
    
    # [END] log
    result = {
        "type": "[END]",
        "task_id": task_id,
        "score": final_score,
        "total_reward": total_reward,
        "steps": step_count,
        "elapsed_seconds": round(elapsed, 2),
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result))
    sys.stdout.flush()
    
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the agent on all 3 tasks and report scores."""
    print("=" * 60, file=sys.stderr)
    print("  Data Cleaning Environment - Baseline Inference", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Model:    {MODEL_NAME}", file=sys.stderr)
    print(f"  API URL:  {API_BASE_URL}", file=sys.stderr)
    print(f"  Env URL:  {ENV_URL}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.stderr.flush()
    
    tasks = ["task_easy", "task_medium", "task_hard"]
    results = []
    
    for task_id in tasks:
        print(f"\n{'-' * 40}", file=sys.stderr)
        print(f"  Running: {task_id}", file=sys.stderr)
        print(f"{'-' * 40}", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            result = run_task(task_id, seed=42)
            results.append(result)
        except Exception as e:
            # This catch is for unexpected errors outside the main run_task loop
            # run_task handles its own reset/loop failures
            print(f"  CRITICAL ERROR running {task_id}: {e}", file=sys.stderr)
            sys.stderr.flush()
            # If run_task didn't finish normally, try to emit a fallback [END]
            results.append({
                "type": "[END]",
                "task_id": task_id,
                "score": 0.0,
                "total_reward": 0.0,
                "steps": 0,
                "error": str(e),
            })
    
    # Final summary (to stderr)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("  BASELINE RESULTS SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for r in results:
        tid = r.get("task_id", "?")
        score = r.get("score", 0.0)
        steps = r.get("steps", 0)
        elapsed = r.get("elapsed_seconds", 0)
        print(f"  {tid:15s}  score={score:.4f}  steps={steps}  time={elapsed:.1f}s", file=sys.stderr)
    
    avg_score = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0
    print(f"\n  Average Score: {avg_score:.4f}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    sys.stderr.flush()


if __name__ == "__main__":
    main()
