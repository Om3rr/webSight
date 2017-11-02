from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime

UPLOAD_FOLDER = os.path.join(os.path.curdir,'tempFiles')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def homepage():
    the_time = "Hello world"
    print(request)
    return render_template("index.html")

# <img src="http://loremflickr.com/600/400">
@app.route('/', methods=['POST'])
def handlePost():
    # check if the post request has the file part
    if(request.form['pass'].lower() != "omer"):
        return redirect(request.url)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        extension = file.filename.rsplit('.')[-1]
        currTime = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]
        link = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s"%(currTime, extension))
        file.save(link)
        return """
        <img src="%s">
        """%(link)

@app.route('/tempFiles/<path:path>')
def send_tempFiles(path):
    return send_from_directory('tempFiles', path)


@app.route('/public/<path:path>')
def sendPublic(path):
    return send_from_directory('public', path)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)