from pyimagesearch.motion_detection.SingleMotionDetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask_socketio import SocketIO, emit
import threading
import argparse
import datetime
import imutils
import time
import cv2
import json
from jsmin import jsmin

DebugMode=False
DataFile="data/data.jsonc"

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

peopleAreWatching = threading.Semaphore(0)
peopleCount = 0

# initialize a flask object
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0, framerate=60).start()
time.sleep(2.0)

@socketio.on('connect')
def my_connect():
	global peopleAreWatching
	global peopleCount

	if peopleCount == 0:
		peopleAreWatching.release()

	peopleCount = peopleCount + 1
	emit('count_change', {'data': peopleCount}, broadcast=True)

@socketio.on('disconnect')
def my_disconnect():
	global peopleAreWatching
	global peopleCount

	peopleCount = peopleCount - 1
	if peopleCount <= 0:
		peopleCount = 0
		peopleAreWatching.acquire()

	emit('count_change', {'data': peopleCount}, broadcast=True)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, peopleAreWatching
	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		peopleAreWatching.acquire(blocking=True)
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=800, inter=cv2.INTER_NEAREST)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		if DebugMode:
			# grab the current timestamp and draw it on the frame
			timestamp = datetime.datetime.now()
			cv2.putText(frame, timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)
			# check to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				if DebugMode:
					cv2.rectangle(frame, (minX, minY), (maxX, maxY),
						(0, 0, 255), 2)

		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()

		peopleAreWatching.release()

def grab_video():
	global vs, outputFrame, lock, peopleAreWatching

	while True:
		peopleAreWatching.acquire(blocking=True)

		frame = vs.read()
		frame = imutils.resize(frame, width=800, inter=cv2.INTER_NEAREST)

		with lock:
			outputFrame = frame.copy()
		time.sleep(1 / 30)

		peopleAreWatching.release()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
	with open(DataFile, "r") as dataFileObj:
		minDataFile = jsmin(dataFileObj.read())
		data = json.loads(minDataFile)

		pageTitle = data["pageTitle"]
		pageHeader = data["pageHeader"]
		styleSheet = data["styleSheet"]
		info = dict(map(lambda di : (di["title"], di["data"]), data["info"]))

		return render_template("index.html",
			pageTitle = pageTitle,
			pageHeader = pageHeader,
			styleSheet = styleSheet,
			info = info)

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	DebugMode

	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	ap.add_argument("-d", "--debug", action='store_true',
		help="Debug mode")
	ap.add_argument("-s", "--stream", action='store_true',
		help="Use direct streaming mode without motion detection. Good for slow systems.")
	ap.add_argument("--datafile", type=str, default="data/data.jsonc",
		help="path to data file (defaults to data/data.jsonc)")
	args = vars(ap.parse_args())

	DebugMode=args["debug"]
	DataFile=args["datafile"]

	if args["stream"]:
		t = threading.Thread(target=grab_video)
		t.daemon = True
		t.start()
	else:
		# start a thread that will perform motion detection
		t = threading.Thread(target=detect_motion, args=(
			args["frame_count"],))
		t.daemon = True
		t.start()

	# start the flask app
	socketio.run(app, host=args["ip"], port=args["port"], debug=DebugMode, use_reloader=DebugMode)

# release the video stream pointer
vs.stop()
