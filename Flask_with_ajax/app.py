#importing libraries
from crypt import methods
from enum import unique
from bson.objectid import ObjectId
from flask import Flask, render_template, request, url_for, redirect
from pymongo import MongoClient 
from bson.json_util import dumps
import json

#flask app cretion
app = Flask(__name__)

#creating connection with mongodb
client = MongoClient('localhost', 27017)

db = client.flask1_db
todos = db.todos

@app.route('/', methods=('GET', 'POST'))
def index():
    
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        try:
            todos.insert_one({'content': content, 'degree': degree})
        except:
            print("Duplicate key")
            return render_template('index.html', z="Todos Already Exist. Add New Todo.")
        return redirect(url_for('index'))


    all_todos = todos.find()
    return render_template('index.html', todos=all_todos, z="")
    

#to edit the document
@app.route('/edit/<string:id>',methods=['POST','GET'])
def edit(id):
    data = todos.find_one({'_id': ObjectId(id)})
    print(data)
    return render_template('edit.html', todos=data)    

#to delete the document
@app.route('/delete/<string:id>',methods=['POST','GET'])
def delete(id):
    todos.delete_one({"_id": ObjectId(id)})
    return redirect(url_for('index'))
   
#to update the document
@app.route('/update/<id>', methods=['POST'])
def update_todos(id):
    if request.method=='POST':
        content1 = request.form['content']
        degree1 = request.form['degree']
        
        try:
            #  todos.insert_one({'content': content1, 'degree': degree1})
            todos.replace_one({"_id": ObjectId(id)},{"_id": ObjectId(id),'content':content1,'degree':degree1})
        except:
             print("Duplicate key")
             return render_template('index.html', z="Todos Already Exist. Add New Todo.")


       
    all_todos = todos.find()
    return redirect(url_for('index'))

#to read the documents in the browser itself.
@app.route('/read',methods = ['GET','POST'])
def read():
    if request.method == 'GET':
        data = todos.find()
        records = []
        for record in data:
            records.append(record)
        data = dumps(records)
        return data 

if __name__=="__main__":
    app.run(debug=True)
