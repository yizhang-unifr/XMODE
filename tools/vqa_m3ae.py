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
