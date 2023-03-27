import flask
from flask import request, jsonify
import hashlib
import json

app = flask.Flask(__name__)

@app.route('/getInfo', methods = ['GET'])
def getInfo():
    # secret = flask.request.headers.get('secret')

    # result = hashlib.sha256(secret.encode())
    # hex = result.hexdigest()
		
    # if hex != "147e3c03d32d8fd51d90860733df3b6d1ba692614de4d6478451900ac783bf21":
    #     return 'your secret is invalid'

    file = open('dataDetection.json','r')

    dataJSON = json.loads(file.read())

    cup_total = 0
    count_330 = 0
    count_600 = 0
    count_1500 = 0
    berat_total = 0

    jsonData = {
        "Botol_330": count_330,
        "Botol_600": count_600,
        "Botol_1500": count_1500,
        "Cup_total": cup_total,
        "Berat_total": berat_total
    }
    jsonObj = json.dumps(jsonData, indent = 5)

    with open("dataDetection.json", "w") as outfile:
        outfile.write(jsonObj)
    
    return jsonify(dataJSON)

if __name__ == '__main__':
	app.run(host = "0.0.0.0", port = 3011)