from flask import Flask, request, render_template,flash ,redirect, send_from_directory,url_for,jsonify, send_file
import google.generativeai as genai
from instagrapi import Client as InstaClient
import yagmail
import schedule
import time
import threading
from serpapi import GoogleSearch
from twilio.rest import Client
import os
import uuid
import numpy as np
import cv2
from dotenv import load_dotenv
import boto3
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from gtts import gTTS
import subprocess
from geopy.geocoders import Nominatim
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)    # Needed for flashing messages

# Configure the generative AI API
def setup_api():
    genai.configure(api_key="AIzaSyAceBlFuOzxRdFdgRmCf9I1lyfB-e-bwM4")
#gemini
def generate_text(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"
#send mail
def send_email(sender_email, receiver_email, subject, message):
    try:
        yag = yagmail.SMTP(sender_email, "kpph grbw kqrh jnud")
        yag.send(
            to=receiver_email,
            subject=subject,
            contents=message
        )
        return "Email has been sent to " + receiver_email
    except Exception as e:
        return f"Failed to send email. Error: {e}"
#scheduling
def schedule_email(sender_email, receiver_email, subject, message, send_time):
    schedule.every().day.at(send_time).do(send_email, sender_email, receiver_email, subject, message)

    while True:
        schedule.run_pending()
        time.sleep(1) 

# Twilio credentials
TWILIO_ACCOUNT_SID = 'ACe0dc0d96549bf7a9790fb8fe1dcc8be0'
TWILIO_AUTH_TOKEN = 'c809b3ed65fb2e1377490817089a6b4f'
TWILIO_PHONE_NUMBER = '+17628008215'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)   
#img filters
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FILTERED_FOLDER'] = 'filtered'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['FILTERED_FOLDER']):
    os.makedirs(app.config['FILTERED_FOLDER'])

def apply_filters(image_path):
    image = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    filters = {
        'Original': image.copy(),
        'Grayscale': image.convert('L'),
        'Blur': image.filter(ImageFilter.BLUR),
        'Gaussian Blur': image.filter(ImageFilter.GaussianBlur(radius=2)),
        'Contour': image.filter(ImageFilter.CONTOUR),
        'Edge Enhance': image.filter(ImageFilter.EDGE_ENHANCE),
        'Sharpen': image.filter(ImageFilter.SHARPEN),
        'Detail': image.filter(ImageFilter.DETAIL),
    }

    sepia = image.copy()
    sepia_data = sepia.getdata()
    sepia_list = []
    for pixel in sepia_data:
        r, g, b = pixel
        tr = int(0.393 * r + 0.769 * g + 0.189 * b)
        tg = int(0.349 * r + 0.686 * g + 0.168 * b)
        tb = int(0.272 * r + 0.534 * g + 0.131 * b)
        sepia_list.append((min(tr, 255), min(tg, 255), min(tb, 255)))
    sepia.putdata(sepia_list)
    filters['Sepia'] = sepia

    enhancer = ImageEnhance.Brightness(image)
    filters['Brightness Enhanced'] = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Contrast(image)
    filters['Contrast Enhanced'] = enhancer.enhance(1.5)

    filters['Posterize'] = ImageOps.posterize(image, bits=2)
    filters['Solarize'] = ImageOps.solarize(image, threshold=128)

    gamma = 1.5
    gamma_corrected = Image.fromarray(np.uint8(255 * (np.array(image) / 255) ** gamma))
    filters['Gamma Corrected'] = gamma_corrected

    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    emboss = cv2.filter2D(cv_image, -1, kernel)
    emboss = Image.fromarray(cv2.cvtColor(emboss, cv2.COLOR_BGR2RGB))
    filters['Emboss'] = emboss

    filtered_images = {}
    for name, img in filters.items():
        filename = f"{name.replace(' ', '_').lower()}_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(app.config['FILTERED_FOLDER'], filename)
        img.save(save_path)
        filtered_images[name] = filename

    return filtered_images

# Your Twilio Account SID and Auth Token
account_sid = 'ACe0dc0d96549bf7a9790fb8fe1dcc8be0'
auth_token = 'c809b3ed65fb2e1377490817089a6b4f'

# Your Twilio WhatsApp number (sandbox or purchased number)
from_whatsapp_number = 'whatsapp:+14155238886'

# EC2 instance 
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = 'ap-south-1'

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials not found. Please set them as environment variables.")

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

ec2 = session.resource('ec2')

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

#routes
#homepage
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
#gemini
@app.route("/generate", methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        prompt = request.form.get("prompt")  # Use .get() to avoid KeyError
        response = generate_text(prompt)
        return render_template("generate.html", prompt=prompt, response=response)
    else:
        return render_template("generate.html")
# Ensure the API is configured
setup_api()
#Send Email instant and scheduling
@app.route('/send_email', methods=['GET', 'POST'])
def email():
    if request.method == 'POST':
        data = request.form
        sender_email = "mohitkumar010305@gmail.com"
        receiver_email = data.get('receiver_email')
        subject = data.get('subject')
        message = data.get('message')
        send_time = data.get('send_time')
        
        if send_time:
            threading.Thread(target=schedule_email, args=(sender_email, receiver_email, subject, message, send_time)).start()
            confirmation_message = f"Email scheduled to be sent at {send_time} to {receiver_email}"
        else:
            confirmation_message = send_email(sender_email, receiver_email, subject, message)
        
        return render_template('send_mail.html', message=confirmation_message)
    else:
        return render_template('send_mail.html')
    
# top 5 search from google
@app.route('/search', methods=['GET', 'POST'])
def search_google():
    if request.method == 'GET':
        query = request.args.get('query')
        results = []
        if query:
            params = {
                "q": query,
                "api_key": "f70544c52980616956503a654345bde97213dafc8593f74ad46ba790729351ed"
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])[:5]
        return render_template('google.html', query=query, results=results)
    else:
        return render_template('google.html',results=[])
# text message
@app.route('/send_message', methods=['GET', 'POST'])
def send_message():
    if request.method == 'POST':
        to_number = request.form['to_number']
        message_body = request.form['message_body']

        try:
            message = client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=to_number
            )
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    return render_template('sms.html')

#imgfilters
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filtered_images = apply_filters(file_path)
            # Ensure filtered_images is a dictionary
            if not isinstance(filtered_images, dict):
                raise TypeError("filtered_images is not a dictionary.")
            return render_template('gallery.html', filtered_images=filtered_images)
    return render_template('imgfilter.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['FILTERED_FOLDER'], filename)

#sending whatsapp message
@app.route('/send_whatsapp', methods=['GET', 'POST'])
def send_whatsapp():
    if request.method == 'POST':
        to_whatsapp_number = request.form['phone_number']
        user_message = request.form['message']
        
        # Validate the input
        if not to_whatsapp_number or not user_message:
            flash('All fields are required!')
            return redirect(url_for('send_whatsapp'))

        to_whatsapp_number = f'whatsapp:{to_whatsapp_number}'

        # Initialize Twilio Client
        client = Client(account_sid, auth_token)

        try:
            # Send message
            message = client.messages.create(
                body=user_message,
                from_=from_whatsapp_number,
                to=to_whatsapp_number
            )
            flash(f'Message sent successfully. SID: {message.sid}','sucess')
        except Exception as e:
            flash(f'Failed to send message: {str(e)}','error')

        return redirect(url_for('send_whatsapp'))

    return render_template('smswhatsapp.html')
#EC2 instances
@app.route("/instances")
def home():
    return render_template("instances.html")

@app.route("/launch-instance", methods=["POST"])
def launch_instance():
    try:
        data = request.json
        instance_type = data.get('instance_type', 't2.micro')
        ami_id = data.get('ami_id', 'ami-0ec0e125bb6c6e8ec')

        instances = ec2.create_instances(
            InstanceType=instance_type,
            ImageId=ami_id,
            MaxCount=1,
            MinCount=1
        )

        instance_ids = [instance.id for instance in instances]
        return jsonify({"message": "Instance launched", "instance_ids": instance_ids})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop-instance", methods=["POST"])
def stop_instance():
    try:
        data = request.json
        instance_id = data.get('instance_id')

        instance = ec2.Instance(instance_id)
        instance.stop()

        return jsonify({"message": f"Instance {instance_id} stopped"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/terminate-instance", methods=["POST"])
def terminate_instance():
    try:
        data = request.json
        instance_id = data.get('instance_id')

        instance = ec2.Instance(instance_id)
        instance.terminate()

        return jsonify({"message": f"Instance {instance_id} terminated"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list-instances")
def list_instances():
    try:
        instances = ec2.instances.all()
        instance_list = []

        for instance in instances:
            instance_list.append({
                "instance_id": instance.id,
                "instance_type": instance.instance_type,
                "state": instance.state['Name'],
                "launch_time": instance.launch_time.strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify({"instances": instance_list})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/texttospeech', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        text = request.form['text']
        # Use gTTS to generate an audio file
        tts = gTTS(text)
        audio_file = "static/audio.mp3"
        tts.save(audio_file)

        return render_template('audio.html', text=text, audio_file=audio_file)
    return render_template('audio.html')

@app.route('/play_audio')
def play_audio():
    # Serve the audio file from the static directory
    return send_file('static/audio.mp3')

@app.route('/download')
def download_audio():
    # Provide a link to download the audio file
    return send_file('static/audio.mp3', as_attachment=True)

#call
@app.route('/make_call', methods=['GET', 'POST'])
def make_call():
    if request.method == 'POST':
        to_phone_number = request.form.get('to')
        message_body = request.form.get('message')

        if not to_phone_number or not message_body:
            return jsonify({'error': 'Both "to" and "message" fields are required'}), 400

        try:
            call = client.calls.create(
                to=to_phone_number,
                from_=TWILIO_PHONE_NUMBER,
                twiml=f'<Response><Say>{message_body}</Say></Response>'
            )
            return jsonify({'message': 'Call initiated successfully', 'call_sid': call.sid}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # GET request handler (rendering the form)
    return render_template('call.html')

#Post Instagram
@app.route("/post_photo", methods=["GET", "POST"])
def instagram_post():
    if request.method == "POST":
        try:
            username = request.form["username"]
            password = request.form["password"]
            photo = request.files["photo"]
            caption = request.form["caption"]

            client = InstaClient()
            client.login(username, password)

            # Save the uploaded photo to the 'uploads' directory
            photo_path = f"uploads/{photo.filename}"
            photo.save(photo_path)

            # Upload the photo to Instagram
            post = client.photo_upload(photo_path, caption)

            # Redirect to the same page with a success message
            return redirect(url_for('instagram_post', success=True))
        except Exception as e:
            return f"An error occurred: {e}", 500

    # Handle GET request or show success alert
    success = request.args.get('success', False)
    return render_template("post_photo.html", success=success)

#docker 

@app.route('/docker')
def docker():
    # Your code here
    return render_template('docker.html')
@app.route("/docker_image_pull", methods=["POST"])
def docker_img_pull():
     image = request.form.get('image')
     cmd = f"docker pull {image}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": "Image downloaded successfully.", "status": "success"})
     else:
         return jsonify({"message": "Image download failed.", "status": "fail"})


@app.route("/launch_docker", methods=["POST"])
def docker_launch():
     container_name = request.form.get('container_name')
     image = request.form.get('image')
     cmd = f"docker run -dit --name {container_name} {image}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": "Docker container launched successfully.", "status": "success", "container_id": output[1]})
     else:
         return jsonify({"message": "Failed to launch Docker container.", "status": "fail"})


@app.route("/docker_stop", methods=["POST"])
def docker_stop():
      container_name = request.form.get('container_name')
      cmd = f"docker stop {container_name}"
      output = subprocess.getstatusoutput(cmd)
      if output[0] == 0:
         return jsonify({"message": "Docker container stopped successfully.", "status": "success"})
      else:
         return jsonify({"message": "Failed to stop Docker container.", "status": "fail"})

@app.route("/docker_start", methods=["POST"])
def docker_start():
     container_name = request.form.get('container_name')
     cmd = f"docker start {container_name}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": "Docker container started successfully.", "status": "success"})
     else:
         return jsonify({"message": "Failed to start Docker container.", "status": "fail"})

@app.route("/docker_status", methods=["POST"])
def docker_status():
     container_name = request.form.get('container_name')
     cmd = f"docker ps -a --filter name={container_name} --format '{{{{.ID}}}} {{{{.Names}}}} {{{{.Status}}}}'"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": output[1], "status": "success"})
     else:
         return jsonify({"message": "Failed to get Docker container status.", "status": "fail"})

@app.route("/docker_remove", methods=["POST"])
def docker_remove():
     container_name = request.form.get('container_name')
     cmd = f"docker rm {container_name}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": "Docker container removed successfully.", "status": "success"})
     else:
         return jsonify({"message": "Failed to remove Docker container.", "status": "fail"})

@app.route("/docker_logs", methods=["POST"])
def docker_logs():
     container_name = request.form.get('container_name')
     cmd = f"docker logs {container_name}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": output[1], "status": "success"})
     else:
         return jsonify({"message": "Failed to get Docker container logs.", "status": "fail"})

@app.route("/docker_image_remove", methods=["POST"])
def docker_img_remove():
     image = request.form.get('image')
     cmd = f"docker rmi -f {image}"
     output = subprocess.getstatusoutput(cmd)
     if output[0] == 0:
         return jsonify({"message": "Docker image removed successfully.", "status": "success"})
     else:
             return jsonify({"message": "Failed to remove Docker image.", "status": "fail"})
#contacts
@app.route('/contact')
def contact():
    return render_template('contact.html')
#about
@app.route('/about')
def about():
    return render_template('about.html')

#geocoordinates
# Initialize the geolocator with a user agent
geolocator = Nominatim(user_agent="geoapp")

@app.route('/geo')
def getLocation():
    return render_template('geo.html')

@app.route('/get_address', methods=['POST'])
def get_address():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    
    if latitude is None or longitude is None:
        return jsonify({'error': 'Invalid coordinates provided.'}), 400

    try:
        # Perform reverse geocoding to get the address
        location = geolocator.reverse((latitude, longitude), exactly_one=True, timeout=10)
        address = location.address if location else 'Address not found.'
    except Exception as e:
        address = f'Error retrieving address: {e}'

    return jsonify({
        'latitude': latitude,
        'longitude': longitude,
        'address': address
    })

# Route to control volume
@app.route('/volume')
def setvolume():
    return render_template('volume.html')

@app.route('/set_volume', methods=['POST'])
def set_volume_route():
    data = request.json
    volume_level = data.get('volume_level')

    # Convert volume_level to float and check if it's valid
    try:
        volume_level = float(volume_level)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid volume level. It must be a number between 0.0 and 1.0."}), 400

    if not (0.0 <= volume_level <= 1.0):
        return jsonify({"error": "Invalid volume level. It must be between 0.0 and 1.0."}), 400

    set_volume(volume_level)
    return jsonify({"status": "Volume set successfully!"})

def set_volume(volume_level):
    # Initialize COM library
    CoInitialize()

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(volume_level, None)
    finally:
        # Uninitialize COM library
        CoUninitialize()
#speech to text
@app.route('/speech')
def speech():
    return render_template('speech_to_text.html')

#camera

@app.route('/camera')
def capture():
    return render_template('camera.html')

@app.route('/live')
def camera():
    return render_template('livestream.html')

if __name__ == "__main__":
    app.run(debug=True)
