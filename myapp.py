from flask import Flask, g, request, jsonify
import sqlite3
import werkzeug
import torch
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
from flask_cors import CORS, cross_origin


app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resorces={r'/*': {"origins": '*'}})


model = torch.load('adam_resnet101_9classes60.pth',map_location=torch.device('cpu'))

def connect_db():
    sql = sqlite3.connect('./database.db')
    sql.row_factory = sqlite3.Row
    return sql

def get_db():
    if not hasattr(g, 'sqlite3'):
        g.sqlite3_db = connect_db()
    return g.sqlite3_db

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route('/update',methods = ['POST'])
def update():
    data = request.get_json()
    count1 = request.json['msg8']
    count2 = request.json['msg9']
    label = None
    if 'label' in data:
        label = request.json['label']
        print(label)
    else:
        # log an error or return an error response
        print("Error: The key 'label' is missing in the request data.")
        return
    print(count1)

    db = get_db()
    c = db.cursor()
    c.execute("UPDATE leaf SET COUNT_RIGHT = ?, COUNT_WRONG = ? where ENGLISH_NAME=?", (count1, count2, label))
    db.commit()
    db.close()

    return jsonify({'result': 'Success'}), 200

def viewdetails(label):
    d = {}
    db = get_db()
    
    cursor = db.execute(f'SELECT * FROM leaf where ENGLISH_NAME="{label}"')
    results = cursor.fetchall()
    d['SCIENTIFIC_NAME'] = results[0]['SCIENTIFIC_NAME']
    d['MALAYALAM_NAME'] = results[0]['MALAYALAM_NAME']
    d['ENGLISH_NAME'] = results[0]['ENGLISH_NAME']
    d['COMMON_NAME'] = results[0]['COMMON_NAME']
    d['USEFUL_PARTS'] = results[0]['USEFUL_PARTS']
    d['MEDICINAL_USE'] = results[0]['MEDICINAL_USE']
    d['PLANT_DESCRIPTION'] = results[0]['PLANT_DESCRIPTION']
    return d

@cross_origin(origin='*', headers=['Content-Type', 'multipart/form-data'])
@app.route('/upload',methods = ['POST'])
def upload():
    if(request.method == 'POST'):
        print("inside post ")
        print(request.files)
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save('./uploadedimages/' + filename)
        print(filename)
        label = get_result('./uploadedimages/' + filename)
        if label:
            decoded = viewdetails(label)
            return jsonify({
                'output1': decoded['SCIENTIFIC_NAME'],
                'output2': decoded['MALAYALAM_NAME'],
                'output3': decoded['ENGLISH_NAME'],
                'output4': decoded['COMMON_NAME'],
                'output5': decoded['USEFUL_PARTS'],
                'output6': decoded['MEDICINAL_USE'],
                'output7': decoded['PLANT_DESCRIPTION'],
                'msg':''
            })
        else:
            return jsonify({
                "msg": "The class is not present"
            })
    
def get_result(file_path):
    '''get the file path and resize the image after that predict using the image'''
    imsize = 256
    label = ''
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    def image_loader(image):
        """load image, returns cuda tensor"""
        image = loader(image).float()
        image = image.unsqueeze(0)
        return image

    img = Image.open(file_path)
    image = image_loader(img)
    with torch.no_grad():
        logits = model.forward(image)
    #ps = torch.exp(logits)
    #_, predTest = torch.max(ps,1)
    #print(ps)
    #print(predTest[0])
    y = torch.softmax(logits, dim=1)
    #print('softmax:',y)
    #print('class:',torch.argmax(y))
    if (torch.max(y)<0.3):
        label = ''
    else:
        indices = torch.argmax(y)
        class_names = ['Chritmas bush','Guava', 'Mimosa','Golden Apple','Indian Goose Berry','Tamarind', 'Curry leaves','Indian aloe','Moringa tree','Neem tree']
        for i in range(len(class_names)):
            if indices == i:
                label = class_names[i]
    return label



if __name__ == '__main__':
    app.run(host="0.0.0.0")
