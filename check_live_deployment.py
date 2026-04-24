import requests
import json
import time

URL = "https://hinex-07-data-cleaning-env.hf.space"

def test_reset():
    print(f"Testing Reset: {URL}/reset")
    payload = {"seed": 42, "task_id": "task_easy"}
    try:
        r = requests.post(f"{URL}/reset", json=payload, timeout=30)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Reset Response OK")
            return r.json().get("observation", {}).get("observation_id")
        else:
            print(f"Reset Failed: {r.text}")
    except Exception as e:
        print(f"Reset Error: {e}")
    return None

def test_step():
    print(f"\nTesting Step: {URL}/step")
    # Action for remove_duplicates
    action = {"action_type": "remove_duplicates", "column": None, "params": {}}
    try:
        r = requests.post(f"{URL}/step", json={"action": action}, timeout=30)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Step Response OK")
            print(f"Reward: {r.json().get('reward')}")
            return True
        else:
            print(f"Step Failed: {r.text}")
    except Exception as e:
        print(f"Step Error: {e}")
    return False

def test_state():
    print(f"\nTesting State: {URL}/state")
    try:
        r = requests.get(f"{URL}/state", timeout=30)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("State Response OK")
            state = r.json()
            print(f"Task ID: {state.get('observation', {}).get('task_id')}")
            print(f"Step Count: {state.get('observation', {}).get('current_step')}")
            return True
        else:
            print(f"State Failed: {r.text}")
    except Exception as e:
        print(f"State Error: {e}")
    return False

if __name__ == "__main__":
    if test_reset():
        if test_step():
            test_state()
