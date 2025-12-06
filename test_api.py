import requests
import json
import time

URL = "http://localhost:8000/predict"

def test_prediction():
    print(f"Testing API at {URL}...")
    
    # Generic features matching the schema
    payload = {
        "feature_1": 0.5,
        "feature_2": -1.2,
        "feature_3": 3.0, # High value might trigger fraud in our synthetic logic
        "feature_4": 0.1,
        "feature_5": -0.5
    }
    
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\n✅ API Response:")
        print(json.dumps(result, indent=2))
        
        if "fraudulent" in result and "fraud_probability" in result:
             print("\nTest passed: Response contains expected fields.")
        else:
             print("\nTest failed: Missing fields in response.")
             
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Is it running?")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Wait a bit if we just started the server
    time.sleep(2)
    test_prediction()
