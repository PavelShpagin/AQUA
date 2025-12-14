import requests

try:
    print(requests.post("https://apigw.dplane.ppgr.io/yukon/api/v0/score-batch/en", 
          json={"texts": ["I like turtles .", "I likes turtles ."]}, verify=False).json())
except Exception as e:
    print(f"Request failed: {e}")