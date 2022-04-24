from bson.objectid import ObjectId
from flask import Flask,render_template, request, url_for, redirect 
#render_template, we use it to render an HTML template.
#the request object to access data the user will submit.
#the url_for to generate URLs.
#redirect to redirect the user back to the index page after adding a todo.
from pymongo import MongoClient

app = Flask(__name__)    #flask app

# MongoClient allows you to connect and interact with your MongoDB server. 
client = MongoClient('localhost', 27017)

db = client.flask_db
todos = db.todos

@app.route('/', methods=('GET', 'POST')) #GET requests are used to retrieve data from the server and POST requests are used to post data to a specific route.
def index():
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        todos.insert_one({'content': content, 'degree': degree}) #insert_one is used on the todos collection to add a todo document into it.
        return redirect(url_for('index'))

    all_todos = todos.find() #To display all the saved todos, you use the find.
    return render_template('index.html', todos=all_todos)
    
#Deleting Todos
@app.post('/<id>/delete/')
def delete(id):
    todos.delete_one({"_id": ObjectId(id)})
    return redirect(url_for('index'))

@app.post('/<id>/edit/')
def edit(id):
    data = todos.find_one({'_id': ObjectId(id)})
    print(data)
    return render_template('edit.html', todos=data)

@app.post('/<id>/update/')
def update_todos(id):
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        todos.replaceOne(
        { "_id": id },
        { "_id": id, "content": content, "degree": degree}
        ) 
       

    all_todos = todos.find()
    return render_template('index', todos=all_todos)
          



