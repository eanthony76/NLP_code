import requests

def speak_api(ip_address, text):
    url = f'http://{ip_address}/predict'
    data = {'text': text}

    response = requests.post(url, json=data)

    if response.status_code==200:
        with open('test.wav', 'wb') as audio_file:
            audio_file.write(response.content)

    else:
        print(f"Error {response.status_code}: {response.text}")
