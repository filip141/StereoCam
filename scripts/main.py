import os
import json
from threading import Lock
from flask import Flask, render_template, Response
from camera import DistanceCamera

mutex = Lock()
objects_list = []
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
app = Flask('object_distance_app', template_folder=template_dir, static_folder=static_dir)


@app.route('/')
def index():
    return render_template("index.html", objects_list=objects_list)


def objects_table(objects):
    global objects_list
    mutex.acquire()
    objects_list = objects
    mutex.release()


@app.route('/get_objects')
def get_objects():
    global objects_list
    mutex.acquire()
    obj_json = json.dumps({'status': 'OK', "objects": objects_list})
    mutex.release()
    return obj_json


def gen(camera):
    while True:
        frame, objects = camera.get_frame()
        objects_table(objects)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(DistanceCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)

