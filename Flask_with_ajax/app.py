from crypt import methods
from bson.objectid import ObjectId
from flask import Flask, render_template, request, url_for, redirect
from pymongo import MongoClient 
from bson.json_util import dumps
import json


app = Flask(__name__)

client = MongoClient('localhost', 27017)

db = client.flask1_db
todos = db.todos

@app.route('/', methods=('GET', 'POST'))
def index():
    # dev = request.get_json()
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        todos.insert_one({'content': content, 'degree': degree})
        return redirect(url_for('index'))

    all_todos = todos.find()
    return render_template('index.html', todos=all_todos)

@app.route('/edit/<string:id>',methods=['POST','GET'])
def edit(id):
    data = todos.find_one({'_id': ObjectId(id)})
    print(data)
    return render_template('edit.html', todos=data)    

@app.route('/delete/<string:id>',methods=['POST','GET'])
def delete(id):
    todos.delete_one({"_id": ObjectId(id)})
    return redirect(url_for('index'))
   
# @app.post('/update/<id>')
@app.route('/update/<id>', methods=['POST'])
def update_todos(id):
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        todos.delete_one({"_id": ObjectId(id)})
        todos.insert_one({'content': content, 'degree': degree})
    all_todos = todos.find()
    return redirect(url_for('index'))

@app.route('/read',methods = ['GET','POST'])
def read():
    if request.method == 'GET':
        data = todos.find()
        records = []
        for record in data:
            records.append(record)
        data = dumps(records)
        return data 


