# reserved for m3ae API client
import requests
import base64
from pathlib import Path

end_point = 'localhost:8080'

def base64coode_from_image_id(image_id):
    root = Path(__file__).parent.parent.resolve()
    images_path = root / 'files'
    print(images_path)
    res = [i for i in images_path.rglob(f"{image_id}.jpg")][0]
    #/home/ubuntu/workspace/XMODE-LLMCompiler/files/p15/p15833469/s57883509/1b0b0385-a72d064d-be1f11ed-a39331d1-dde8f464.jpg
    base64code = "data:image/png;base64," + base64.b64encode(res.read_bytes()).decode("ascii")
    return base64code
    

def post_vqa_m3ae(question, image_id):
    url = f"http://{end_point}/predict"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'question': question,
        'image_url': base64coode_from_image_id(image_id)
    }
    response = requests.post(url, data=data, headers=headers)
    return response.json()

def post_vqa_m3ae_with_url(question, image_url):
    url = f"http://{end_point}/predict"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    data = {
        'question': question,
        'image_url': image_url
    }
    # print(data, image_url)
    response = requests.post(url, data=data, headers=headers)
    return response.json()

