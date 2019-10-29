from flask import Flask, render_template, request, send_file

app = Flask(__name__)

@app.route("/")
def reload():
	return render_template("index.html")

@app.route('/get_match', methods=['POST'])
def get_match():
	pic = request.form['pic']
	print("pic: " + str(pic))
	return send_file(pic, mimetype='image')

if __name__ == "__main__":
	app.run(debug=True, port=5000)