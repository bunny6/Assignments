from flask import Flask, render_template, request, jsonify #imported flask.

app = Flask(__name__)

@app.route('/')    #route will tell flask what URL should be triggered.
def index():
	return render_template('form.html') #render_template helps function to serve an HTML template as the response.

@app.route('/process', methods=['POST']) #POST will post the data to specific route.
def process():

	email = request.form['email']
	name = request.form['name']

	if name and email:
		newName = name[::]

		return jsonify({'name' : newName})   #returns the json object

	return jsonify({'error' : 'Missing data!'})   #if name or email is missing, it will return error.

if __name__ == '__main__':
	app.run(debug=True)