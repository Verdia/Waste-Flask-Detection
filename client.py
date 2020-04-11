import numpy as np
import requests
import json
import cv2

address = 'http://localhost:5000'
url = address + '/object_detection'
img_path = 'images/girl_image.jpg'
save_dir = "outputs/"

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

def post_image(img_path, URL):
    """ post image and return the response """
 
    img = open(img_path, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response

def parse_response(response):
    return json.loads(response.text)

def write_image(save_dir, parsed_response):
    print("the image is being written in {}".format(save_dir))
    img_array = np.asarray(parsed_response["image data"])
    cv2.imwrite(save_dir + "server_output.jpg", img_array)
    return img_array

response = post_image(img_path, url)
parsed_response = parse_response(response)

print("detections are: ", parsed_response["detections"])
print("the image size is: ", parsed_response["image size"])

write_image(save_dir, parsed_response)
