import flask
from flask import Flask, render_template, jsonify, send_from_directory
from flask import request
from flask_cors import CORS
import redis
import datetime

currentPage = "0"

app = Flask(__name__)
CORS(app)
red = redis.StrictRedis()


@app.route('/')
def index():
    return 'Index Page'


@app.route('/hello')
def hello():
    return 'Hello, World'


def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('chat')
    # TODO: handle client disconnection.
    for message in pubsub.listen():
        print(message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'POST':
        flask.session['user'] = flask.request.form['user']
        return flask.redirect('/')
    return '<form action="" method="post">user: <input name="user">'


@app.route('/post', methods=['POST'])
def post():
    message = flask.request.form['message']
    user = flask.session.get('user', 'anonymous')
    now = datetime.datetime.now().replace(microsecond=0).time()
    red.publish('chat', u'[%s] %s: %s' % (now.isoformat(), user, message))
    return flask.Response(status=204)


@app.route('/stream')
def stream():
    return flask.Response(event_stream(),
                          mimetype="text/event-stream")


@app.route('/getText')
def get_text():
    with open("./static/status.txt", 'r') as outfile:
        global currentPage
        status = outfile.readline()
    if status == "Next":
        currentPage = "menuOptions"
    else:
        currentPage = "getText"
    return status


@app.route('/menuOptions')
def menu():
    global currentPage
    with open("./static/menu_option.txt", 'r') as outfile:
        option = outfile.readline()
    if option == "1" or option == "2":
        currentPage = "dispenseStatus"
    else:
        currentPage = "menuOptions"
    return option


@app.route('/dispenseStatus')
def water_dispense_status():
    global currentPage
    currentPage = "resultPage"
    with open("./static/drinks_dispense_status.txt", 'r') as outfile:
        dispense_status = outfile.readline()
    if dispense_status == "Next":
        currentPage = "resultPage"
    else:
        currentPage = "dispenseStatus"
    return dispense_status


@app.route('/resultPage')
def result_page():
    global currentPage
    currentPage = "horribleImage"
    with open("./static/result_page.txt", 'r') as outfile:
        result_page = outfile.readline()
    if result_page == "Next":
        currentPage = "horribleImage"
    else:
        currentPage = "resultPage"
    return result_page


@app.route('/horribleImage')
def horrible_image():
    global currentPage
    currentPage = "0"
    with open("./static/horrible_image.txt", 'r') as outfile:
        horrible_image = outfile.readline()
    if result_page == "Next":
        currentPage = "0"
    else:
        currentPage = "horribleImage"
    return horrible_image


@app.route('/control', methods=['GET', 'POST'])
def control():
    return render_template('control.html')


@app.route('/option1')
def option1():
    if currentPage == "menuOptions":
        with open("./static/menu_option.txt", 'w') as outfile:
            outfile.write("1")
    return


@app.route('/option2')
def option2():
    if currentPage == "menuOptions":
        with open("./static/menu_option.txt", 'w') as outfile:
            outfile.write("2")
    return


@app.route('/next')
def next():
    if currentPage == "getText":
        # with open("./static/drinks_dispense_status.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/result_page.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/horrible_image.txt", 'w') as outfile:
        #     outfile.write("")
        with open("./static/horrible_image.txt", 'w') as outfile:
            outfile.write("")
        with open("./static/status.txt", 'w') as outfile:
            outfile.write("Next")
    elif currentPage == "dispenseStatus":
        # with open("./static/status.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/horrible_image.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/result_page.txt", 'w') as outfile:
        #     outfile.write("")
        with open("./static/menu_option.txt", 'w') as outfile:
            outfile.write("")
        with open("./static/drinks_dispense_status.txt", 'w') as outfile:
            outfile.write("Next")
    elif currentPage == "resultPage":
        # with open("./static/status.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/horrible_image.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/menu_option.txt", 'w') as outfile:
        #     outfile.write("")
        with open("./static/drinks_dispense_status.txt", 'w') as outfile:
            outfile.write("")
        with open("./static/result_page.txt", 'w') as outfile:
            outfile.write("Next")
    elif currentPage == "horribleImage":
        with open("./static/status.txt", 'w') as outfile:
            outfile.write("")
        # with open("./static/drinks_dispense_status.txt", 'w') as outfile:
        #     outfile.write("")
        # with open("./static/menu_option.txt", 'w') as outfile:
        #     outfile.write("")
        with open("./static/result_page.txt", 'w') as outfile:
            outfile.write("")
        with open("./static/horrible_image.txt", 'w') as outfile:
            outfile.write("Next")
        return jsonify(currentPage)


if __name__ == "__main__":
    # app.run(host= '0.0.0.0', port=8800)
    app.run(host='192.168.8.100', port=5000, debug=False)
    # app.run(host= '10.27.68.170', port=8800, debug=False)
