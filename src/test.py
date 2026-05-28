import json

import requests

url = "https://mlops-it7t.onrender.com/predict"

x_new = dict(
    {
        "ph": 7.0,
        "Hardness": 150.0,
        "Solids": 50000.0,
        "Chloramines": 7.0,
        "Sulfate": 300.0,
        "Conductivity": 500.0,
        "Organic_carbon": 10.0,
        "Trihalomethanes": 80.0,
        "Turbidity": 5.0,
    }
)

response = requests.post(url, json=x_new)

print("Response Text:", response.text)
print("Status Code:", response.status_code)
