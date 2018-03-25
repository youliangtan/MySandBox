# app = Flask(__name__)
from flask import Flask, render_template, request, jsonify, send_from_directory
import time

app = Flask(__name__, static_url_path='')

# @app.route('/pics/<path:path>')
# def get_pics(path):
#     print(path)
#     return send_from_directory('/templates/pics/', path)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#delete will go wrong
@app.route('/templates/<path:filename>', methods  = ['GET', 'POST'])
def send_pics(filename):
    return send_from_directory('/', filename)

# url_for('pics', filename='breakfast.png')


# @app.route('/pics/<path:path>')
# def static_file(path):
#     print(path)
#     return app.send_static_file('./templates/pics/' + path)

# @app.route('/<path:filename>')
# def static_proxy(path):
#     print(path)
#     print("jibai")
#   #d_static_file will guess the correct MIME type
#     return app.send_static_file(path)


# @app.route('/home/<path:path>')
# def send_command(path):
#     print(send_from_directory('/templates/pics', path))
#     return send_from_directory('/templates/pics', path)

@app.route('/', methods  = ['GET', 'POST'])
def home():
    # if request.method =='POST':
    #     num = request.form['num']

    #     selection = request.form['selection']
    #     print("yes, input is received!")
    with open("./static/choice.txt", 'w') as outfile:
        #create empty string 
        outfile.write('')    
    print("Reminder: Run it in the right working Dir as python code")
    return render_template('index.html')

@app.route('/smile', methods  = ['GET', 'POST'])
def smile():
    with open("./static/smile.txt", 'w') as outfile:
        #create empty string 
        outfile.write('smiled')      
    return render_template('display.html')

@app.route('/display', methods  = ['GET', 'POST'])
def display():
    return render_template('display.html')

@app.route('/send')
def send():
    eggNum = request.args.get('eggNum', 0, type=str)
    toastNum = request.args.get('toastNum', 0, type=str)
    drink = request.args.get('drink', 0, type=str)

    printout = "{}, {} Toast, {} Egg".format(drink, toastNum, eggNum)
    print(printout)
    textinput = "{},{},{}.".format(drink[0], toastNum, eggNum)
    with open("./static/choice.txt", 'w') as outfile:
        outfile.write(textinput)
    with open("./static/smile.txt", 'w') as outfile:
        outfile.write("pending")
    # if request.method =='POST':
    # render_template('display.html')
    return jsonify(result=printout)


if __name__ == "__main__":
    # app.run(host= '0.0.0.0', port=5000)
    # app.run(host= '192.168.31.160', port=8888, debug=False)
    # app.run(host= '10.27.240.15', port=8080, debug=False)
    app.run(host= '192.168.1.2', port=8000, debug=False)
