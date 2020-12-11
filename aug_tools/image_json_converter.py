import sys
import base64
import json
from base64 import b64encode


def image_to_json(image_path, json_name, encoding_type = 'utf-8'):
   # read image with binary, get origin byte data, notice is 'rb'
   with open(image_path, 'rb') as jpg_file:
       byte_content = jpg_file.read()

   # encode to base64
   base64_bytes = b64encode(byte_content)

   # decode base64 to utf-8
   base64_string = base64_bytes.decode(encoding_type)

   ## save data as dictionary
   raw_data = {}
   raw_data["name"] = image_path
   raw_data["image_base64_string"] = base64_string

   ## save data as json, and 2 indent
   json_data = json.dumps(raw_data, indent=2)
   with open(json_name, 'w') as json_file:
       json_file.write(json_data)

def json_to_image(json_name, image_path):
    with open(json_name, "r") as json_file:
        raw_data = json.load(json_file)
    image_base64_string = raw_data["image_base64_string"]
    image_data = base64.b64decode(image_base64_string)
    with open(image_path, 'wb') as jpg_file:
        jpg_file.write(image_data)
if __name__ == '__main__':
    json_name = 'test.json'
    image_path = './instance.jpg'
    image_to_json(image_path, json_name)
    #json_to_image(json_name, image_path)
