import requests

url = "http://127.0.0.1:5000/detect"
files = {"image": open("npr.brightspotcdn.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
