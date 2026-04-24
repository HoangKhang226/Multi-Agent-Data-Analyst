import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_full_flow():
    print("=== Automated E2E System Test ===")
    
    # 1. Ingest CSV
    print("\n[Step 1] Ingesting CSV...")
    files = {'files': ('test_data.csv', open('data/raw/test_data.csv', 'rb'), 'text/csv')}
    res = requests.post(
        f"{API_BASE_URL}/ingest",
        files=files,
        params={"embedding_provider": "gemini"}
    )
    if res.status_code == 200:
        print("✅ Ingest success.")
        print(f"   Summary: {res.json()[0].get('summary')[:100]}...")
    else:
        print(f"❌ Ingest failed: {res.text}")
        return

    # 2. Check File Structure
    print("\n[Step 2] Checking filesystem (manually via logs/script)...")
    # This step is done by the agent using list_dir later
    
    # 3. Chat with Data
    print("\n[Step 3] Chatting to verify Routing...")
    payload = {
        "question": "Vẽ biểu đồ hình cột cho Sales",
        "llm_provider": "gemini",
        "user_id": "tester_bot",
        "data_mode": "tabular",
        "retrieval_mode": "hierarchical"
    }
    
    start_time = time.time()
    res = requests.post(f"{API_BASE_URL}/chat", json=payload)
    duration = time.time() - start_time
    
    if res.status_code == 200:
        print(f"✅ Chat success in {duration:.2f}s.")
        answer = res.json()["answer"]
        print(f"   Assistant: {answer[:200]}...")
        if "charts" in res.json() and res.json()["charts"]:
            print(f"   📈 Charts generated: {res.json()['charts']}")
    else:
        print(f"❌ Chat failed: {res.text}")

if __name__ == "__main__":
    try:
        test_full_flow()
    except Exception as e:
        print(f"🚨 Connection Error: {e}. Is the backend running at {API_BASE_URL}?")
