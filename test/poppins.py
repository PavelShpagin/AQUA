#!/usr/bin/env python3
import requests
import json

def test_poppins_api():
    """Test the new Poppins API endpoint with source/target comparison"""
    
    # Test cases
    test_cases = [
        {
            "source": "Collecting fees at the station to prevent delay caused by passengers lining up to pay the fare on board",
            "target": "Collect fees at the station to prevent delay caused by passengers lining up to pay the fare on board",
            "description": "Word change (turtles -> apples)"
        },
        {
            "source": "This can toy with a person and change emotions or even mood ( Judge & Robbins, p. 98 )",
            "target": "This can toy with a person's emotions or even mood ( Judge & Robbins, p. 98 )",
            "description": "FP1 case: Ward name change"
        },
        {
            "source": "Test failed. See  exception logs above.",
            "target": "Test failed. See the exception logs above.",
            "description": "TP case: Good grammar correction"
        }
    ]
    
    print("Testing new Poppins API endpoint...")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print(f"Source: {case['source']}")
        print(f"Target: {case['target']}")
        
        try:
            response = requests.post(
                "http://poppins.prod-text-processing.grammarlyaws.com/api/v0/process",
                json={
                    "source": case["source"],
                    "target": case["target"]
                },
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                # Try to extract useful comparison metrics
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")
        
        print("-" * 40)
    
    print("\nCurl equivalent for manual testing:")
    print("curl http://poppins.prod-text-processing.grammarlyaws.com/api/v0/process \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"source\": \"I like turtles .\", \"target\": \"I like apples .\"}'")

if __name__ == "__main__":
    test_poppins_api()