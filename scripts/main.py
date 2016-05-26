#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import time
from threading import Lock
from flask import Flask, render_template, Response, request
from camera import DistanceCamera

mutex = Lock()
mutex_param = Lock()
objects_list = []
cam_obj = None
# Distance meter settings
cam_one = "/dev/video1"
cam_two = "/dev/video2"
n_obj = 4
h_param = 5.1

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
app = Flask('app', template_folder=template_dir, static_folder=static_dir)


@app.route('/')
def index():
    return render_template("index.html", objects_list=objects_list)


def objects_table(objects):
    global objects_list
    mutex.acquire()
    objects_list = objects
    mutex.release()


@app.route('/change_camera', methods=['POST'])
def change_camera():
    mutex_param.acquire()
    global cam_two, cam_one, h_param, n_obj
    # Save new camera number one
    new_cam_one = request.json.get("devOne")
    if new_cam_one:
        cam_one = new_cam_one
    # Save new camera number two
    new_cam_two = request.json.get("devTwo")
    if new_cam_two:
        cam_two = new_cam_two
    # Save new nobj parameter
    new_nobj = request.json.get("nObj")
    if new_nobj:
        n_obj = int(new_nobj)
    # Save new horizont param
    new_hor = request.json.get("horizont")
    if new_hor:
        h_param = float(new_hor)
    mutex_param.release()
    return json.dumps({'status': 'OK'})


@app.route('/get_objects')
def get_objects():
    global objects_list
    mutex.acquire()
    obj_json = json.dumps({'status': 'OK', "objects": objects_list})
    mutex.release()
    return obj_json


def gen(camera, nobj, hparam):
    while True:
        frame, objects = camera.get_frame(nobj, hparam)
        objects_table(objects)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    mutex_param.acquire()
    global cam_one, cam_two, n_obj, h_param, cam_obj
    if cam_obj:
        del cam_obj
        time.sleep(5)
    cam_obj = DistanceCamera(cam_one, cam_two)
    return_param = Response(gen(cam_obj, n_obj, h_param),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    mutex_param.release()
    return return_param

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)


