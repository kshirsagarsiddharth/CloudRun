import requests 


response = requests.post("http://localhost:5000/", files={'file': open('three.png','rb')})
print(response.json())