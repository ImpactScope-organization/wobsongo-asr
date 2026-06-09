import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Simple payload (Use short audio for pure testing)
PAYLOAD = {
    "input": {
        # Example using a public audio URL for simplicity
        "audio_url": "https://raw.githubusercontent.com/pdx-cs-sound/wavs/master/voice.wav", 
        "model": "omniASR_LLM_3B",
        "language": "auto"
    }
}

def measure_request_time(test_name: str):
    print(f"\n--- Starting Test: {test_name} ---")
    start_time = time.perf_counter()
    
    try:
        # 1. Send Request
        run_url = f"{BASE_URL}/run"
        resp = requests.post(run_url, headers=HEADERS, json=PAYLOAD)
        resp.raise_for_status()
        job_id = resp.json().get("id")
        print(f"Job submitted! ID: {job_id}. Waiting for processing...")

        # 2. Polling Status
        status_url = f"{BASE_URL}/status/{job_id}"
        while True:
            status_resp = requests.get(status_url, headers=HEADERS)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                # Record time when fully completed
                end_time = time.perf_counter()
                total_time = end_time - start_time
                
                # Fetch internal metrics from RunPod server
                delay_time_server = status_data.get("delayTime", 0) / 1000 # convert ms to seconds
                exec_time_server = status_data.get("executionTime", 0) / 1000 # convert ms to seconds
                
                print(f"Completed!")
                print(f"Total Client Time (Round-trip): {total_time:.2f} seconds")
                print(f"Server Metrics -> Delay/Queue: {delay_time_server:.2f}s | Execution: {exec_time_server:.2f}s")
                break
            elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                print(f"❌ Failed with status: {status}")
                break
            
            time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("MAKE SURE ACTIVE WORKERS IN RUNPOD IS 0 BEFORE STARTING!")
    input("Press Enter if it is confirmed to be 0 (machine is asleep)...")
    
    # 1. Measure Cold Start
    measure_request_time("COLD START (Machine starting from scratch)")
    
    # Give a short pause to let the server breathe
    time.sleep(3) 
    
    # 2. Measure Warm Start (Because the worker is now definitely running / Idle)
    measure_request_time("WARM START (Machine is already running)")