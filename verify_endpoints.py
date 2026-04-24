import requests
import json

BASE_URL = "https://hinex-07-data-cleaning-env.hf.space"

def run_verify():
    print(f"--- Verifying Live Deployment: {BASE_URL} ---")
    
    # 1. RESET
    print("\n1. Testing /reset...")
    reset_payload = {"seed": 42, "task_id": "task_easy"}
    r = requests.post(f"{BASE_URL}/reset", json=reset_payload, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print("SUCCESS: Reset endpoint is responsive.")
        
    # 2. STEP
    print("\n2. Testing /step...")
    step_payload = {"action": {"action_type": "remove_duplicates", "column": None, "params": {}}}
    r = requests.post(f"{BASE_URL}/step", json=step_payload, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print(f"SUCCESS: Step endpoint works. Reward: {res.get('reward')}")
        
    # 3. STATE
    print("\n3. Testing /state...")
    r = requests.get(f"{BASE_URL}/state", timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        step = res.get("observation", {}).get("current_step")
        print(f"SUCCESS: State endpoint works. Current Step: {step}")

if __name__ == "__main__":
    try:
        run_verify()
    except Exception as e:
        print(f"\nERROR: Verfication failed: {e}")
