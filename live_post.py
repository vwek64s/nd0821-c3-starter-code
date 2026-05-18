import requests

url = "https://nd0821-c3-starter-code-c05c.onrender.com/model/"

data = {
    "age": 52,
    "workclass": "Private",
    "fnlgt": 209642,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

response = requests.post(url, json=data)

print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")