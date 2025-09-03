import os
from tensorflow.keras.models import load_model
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import requests
import random  # This line is kept for compatibility with existing code
from datetime import datetime, timedelta
import threading
import time
import pandas as pd

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from werkzeug.utils import secure_filename
from PIL import Image
import random
from werkzeug.utils import secure_filename
from PIL import Image
import random

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///irrigation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


# Route nhận diện sâu bệnh từ ảnh (sửa chuẩn Flask, chỉ dùng model DCNN)
@app.route('/disease_detect', methods=['GET', 'POST'])
def disease_detect():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result = 'Không tìm thấy file ảnh.'
        else:
            file = request.files['image']
            if file.filename == '':
                result = 'Vui lòng chọn file ảnh.'
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join('static', filename)
                file.save(filepath)
                try:
                    disease_model = load_model('model_disease.h5')
                    # Lấy nhãn bệnh theo thứ tự thư mục khi huấn luyện
                    disease_labels = ['Sâu cuốn lá', 'Bệnh đạo ôn', 'Bệnh vàng lá', 'Không phát hiện bệnh']
                    img = Image.open(filepath).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    preds = disease_model.predict(img_array)
                    pred_idx = np.argmax(preds[0])
                    result = disease_labels[pred_idx]
                except Exception as e:
                    result = f'Lỗi xử lý ảnh: {e}'
    return render_template('disease_detect.html', result=result)

# Thông tin Adafruit IO
AIO_USERNAME = "dadanganh"
AIO_KEY = os.getenv('AIO_KEY', 'aio_vedb77WNl6BDiO5heWIvkIsClQe3')
AIO_HEADERS = {'X-AIO-Key': AIO_KEY}

# Load mô hình và các tệp hỗ trợ
try:
    model = joblib.load("ensemble_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder_soil = joblib.load("encoder_soil.pkl")
    encoder_crop = joblib.load("encoder_crop.pkl")
    logging.info("Đã tải mô hình và các tệp hỗ trợ thành công")
except FileNotFoundError as e:
    logging.error(f"Không tìm thấy file: {e}. Vui lòng kiểm tra các file mô hình!")
    model = None
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình hoặc tệp hỗ trợ: {e}")
    model = None

feature_columns = ["Temparature", "Humidity", "Moisture", "SoilType", "CropType"]

# Mô hình User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='user')

# Mô hình SensorData
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    soilmoisture = db.Column(db.Float, nullable=True)
    pump_state = db.Column(db.String(10), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

# Mô hình Configuration
class Configuration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    soil_min = db.Column(db.Float, default=40.0)
    soil_max = db.Column(db.Float, default=60.0)
    auto_mode = db.Column(db.String(10), default='OFF')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Mô hình Schedule
class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scheduled_time = db.Column(db.DateTime, nullable=False)
    off_time = db.Column(db.DateTime, nullable=False)
    executed = db.Column(db.Boolean, default=False)
    repeat_daily = db.Column(db.Boolean, default=False)

def send_feed_data(feed, value):
    try:
        url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/{feed}/data"
        payload = {'value': value}
        response = requests.post(url, headers=AIO_HEADERS, json=payload)
        response.raise_for_status()
        logging.info(f"Đã gửi {value} tới feed {feed}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi gửi dữ liệu tới {feed}: {str(e)}")
        return False

def check_schedules():
    while True:
        with app.app_context():
            now = datetime.now()
            schedules = Schedule.query.filter_by(executed=False).all()
            logging.debug(f"Tìm thấy {len(schedules)} lịch chưa thực hiện")
            for schedule in schedules:
                if now >= schedule.scheduled_time and now < schedule.off_time:
                    send_feed_data('relaycontrol', 'ON')
                    logging.info(f"Bật bơm lúc {schedule.scheduled_time} cho user {schedule.user_id}")
                elif now >= schedule.off_time:
                    success = send_feed_data('relaycontrol', 'OFF')
                    if success:
                        schedule.executed = True
                        db.session.commit()
                        logging.info(f"Đã tắt bơm lúc {schedule.off_time} cho user {schedule.user_id}")
                        if schedule.repeat_daily:
                            new_schedule = Schedule(
                                user_id=schedule.user_id,
                                scheduled_time=schedule.scheduled_time + timedelta(days=1),
                                off_time=schedule.off_time + timedelta(days=1),
                                executed=False,
                                repeat_daily=True
                            )
                            db.session.add(new_schedule)
                            db.session.commit()
                            logging.info(f"Tạo lịch lặp lại mới cho ngày {new_schedule.scheduled_time}")
                    else:
                        logging.error(f"Thất bại khi tắt bơm lúc {schedule.off_time}")
        time.sleep(10)

def check_auto_mode():
    while True:
        with app.app_context():
            # Lấy tất cả người dùng có chế độ tự động bật
            configs = Configuration.query.filter_by(auto_mode='ON').all()
            logging.debug(f"Tìm thấy {len(configs)} người dùng có chế độ tự động bật")
            for config in configs:
                user = User.query.get(config.user_id)
                if not user:
                    logging.warning(f"Không tìm thấy user với ID {config.user_id}")
                    continue

                # Lấy dữ liệu độ ẩm đất mới nhất từ Adafruit IO
                try:
                    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/soilmoisture/data/last"
                    response = requests.get(url, headers=AIO_HEADERS)
                    response.raise_for_status()
                    soil_moisture = float(response.json()['value'])
                    logging.debug(f"Độ ẩm đất của user {user.id}: {soil_moisture}%")
                except (requests.exceptions.RequestException, ValueError) as e:
                    logging.error(f"Lỗi lấy dữ liệu độ ẩm đất cho user {user.id}: {str(e)}")
                    continue

                # Lấy trạng thái bơm hiện tại từ Adafruit IO
                try:
                    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/relaycontrol/data/last"
                    response = requests.get(url, headers=AIO_HEADERS)
                    response.raise_for_status()
                    pump_state = response.json()['value']
                    logging.debug(f"Trạng thái bơm của user {user.id}: {pump_state}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Lỗi lấy trạng thái bơm cho user {user.id}: {str(e)}")
                    pump_state = 'OFF'

                # Kiểm tra và điều khiển relay
                if soil_moisture < config.soil_min and pump_state != 'ON':
                    success = send_feed_data('relaycontrol', 'ON')
                    if success:
                        logging.info(f"Độ ẩm đất ({soil_moisture}%) thấp hơn mục tiêu ({config.soil_min}%), bật bơm tự động cho user {user.id}")
                        # Cập nhật trạng thái bơm trong cơ sở dữ liệu
                        latest_sensor = SensorData.query.filter_by(user_id=user.id).order_by(SensorData.timestamp.desc()).first()
                        if latest_sensor:
                            latest_sensor.pump_state = 'ON'
                            db.session.commit()
                    else:
                        logging.error(f"Thất bại khi bật bơm tự động cho user {user.id}")
                elif soil_moisture >= config.soil_max and pump_state != 'OFF':
                    success = send_feed_data('relaycontrol', 'OFF')
                    if success:
                        logging.info(f"Độ ẩm đất ({soil_moisture}%) vượt mục tiêu ({config.soil_max}%), tắt bơm tự động cho user {user.id}")
                        # Cập nhật trạng thái bơm trong cơ sở dữ liệu
                        latest_sensor = SensorData.query.filter_by(user_id=user.id).order_by(SensorData.timestamp.desc()).first()
                        if latest_sensor:
                            latest_sensor.pump_state = 'OFF'
                            db.session.commit()
                    else:
                        logging.error(f"Thất bại khi tắt bơm tự động cho user {user.id}")
                else:
                    logging.debug(f"Không cần điều chỉnh bơm cho user {user.id}: soil_moisture={soil_moisture}%, soil_min={config.soil_min}%, soil_max={config.soil_max}%, pump_state={pump_state}")
        time.sleep(10)  # Kiểm tra mỗi 10 giây

# Hàm bảo vệ tuyến đường
def login_required(f):
    def wrap(*args, **kwargs):
        if 'username' not in session:
            logging.warning("Chưa đăng nhập, chuyển hướng đến trang login")
            return redirect(url_for('login'))
        user = User.query.filter_by(username=session['username']).first()
        if user is None:
            logging.warning(f"Người dùng {session['username']} không tồn tại trong cơ sở dữ liệu, đăng xuất và chuyển hướng đến trang login")
            session.pop('username', None)
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# Hàm kiểm tra quyền admin
def admin_required(f):
    def wrap(*args, **kwargs):
        if 'username' not in session:
            logging.warning("Chưa đăng nhập, chuyển hướng đến trang login")
            return redirect(url_for('login'))
        user = User.query.filter_by(username=session['username']).first()
        if user is None:
            logging.warning(f"Người dùng {session['username']} không tồn tại trong cơ sở dữ liệu, đăng xuất và chuyển hướng đến trang login")
            session.pop('username', None)
            return redirect(url_for('login'))
        logging.debug(f"Kiểm tra quyền admin - User: {user.username if user else 'None'}, Role: {user.role if user else 'None'}")
        if user.role != 'admin':
            logging.error("Không có quyền admin, trả về lỗi 403")
            flash('Yêu cầu quyền admin!', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# Route tạo tài khoản admin
@app.route('/create-admin/<secret>', methods=['GET'])
def create_admin(secret):
    admin_secret = os.getenv('ADMIN_SECRET', 'admin-secret-2025')
    if secret != admin_secret:
        logging.error(f"Mật khẩu bí mật không đúng: {secret}")
        flash('Mật khẩu bí mật không đúng!', 'error')
        return redirect(url_for('login'))
    admin = User.query.filter_by(username='admin').first()
    if admin:
        logging.info("Tài khoản admin đã tồn tại")
        flash('Tài khoản admin đã tồn tại!', 'error')
        return redirect(url_for('login'))
    try:
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin_user = User(username='admin', password=hashed_password, role='admin')
        db.session.add(admin_user)
        db.session.commit()
        logging.info("Tài khoản admin đã được tạo thành công")
        flash('Tài khoản admin đã được tạo thành công! Đăng nhập với username: admin, password: admin123', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Lỗi khi tạo tài khoản admin: {str(e)}")
        flash(f'Lỗi khi tạo tài khoản admin: {str(e)}', 'error')
    return redirect(url_for('login'))

# Route đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')
        secret = request.form.get('secret', '')

        if not username or not password or not confirm_password:
            logging.error("Thiếu thông tin bắt buộc trong form đăng ký")
            return render_template('auth.html', message="Vui lòng điền đầy đủ thông tin", container_active=(role == 'user'), container_admin_active=(role == 'admin'))

        if password != confirm_password:
            logging.error("Mật khẩu và xác nhận mật khẩu không khớp")
            return render_template('auth.html', message="Mật khẩu không khớp", container_active=(role == 'user'), container_admin_active=(role == 'admin'))

        if User.query.filter_by(username=username).first():
            logging.error(f"Tên người dùng đã tồn tại: {username}")
            return render_template('auth.html', message="Tên người dùng đã tồn tại", container_active=(role == 'user'), container_admin_active=(role == 'admin'))

        if role == 'admin':
            admin_secret = os.getenv('ADMIN_SECRET', 'admin-secret-2025')
            if secret != admin_secret:
                logging.error(f"Mật khẩu bí mật không đúng cho admin: {secret}")
                return render_template('auth.html', message="Mật khẩu bí mật không đúng cho admin", container_active=False, container_admin_active=True)

        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(username=username, password=hashed_password, role=role)
            db.session.add(new_user)
            db.session.commit()
            new_config = Configuration(user_id=new_user.id)
            db.session.add(new_config)
            db.session.commit()
            logging.info(f"Tài khoản {role} đã được tạo thành công: {username}")
            flash(f'Tài khoản {role} đã được tạo thành công!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Lỗi khi tạo tài khoản: {str(e)}")
            return render_template('auth.html', message=f"Lỗi khi tạo tài khoản: {str(e)}", container_active=(role == 'user'), container_admin_active=(role == 'admin'))
    return render_template('auth.html', container_active=False, container_admin_active=False)

# Route đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'user')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            if user.role != role:
                logging.error(f"Vai trò không khớp: yêu cầu {role}, thực tế {user.role}")
                return render_template('auth.html', message=f"Tài khoản này không phải {role}", container_active=(role == 'user'), container_admin_active=(role == 'admin'))
            session['username'] = username
            logging.info(f"Đăng nhập thành công: {username} (vai trò: {role})")
            flash('Đăng nhập thành công!', 'success')
            if user.role == 'admin':
                logging.debug(f"Chuyển hướng tài khoản admin {username} đến trang admin")
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        logging.error(f"Đăng nhập thất bại: Tên người dùng hoặc mật khẩu không đúng - {username}")
        return render_template('auth.html', message="Tên người dùng hoặc mật khẩu không đúng", container_active=(role == 'user'), container_admin_active=(role == 'admin'))
    return render_template('auth.html', container_active=True, container_admin_active=False)

# Route đăng xuất
@app.route('/logout')
def logout():
    session.pop('username', None)
    logging.info("Đăng xuất thành công")
    flash('Đăng xuất thành công!', 'success')
    return redirect(url_for('login'))

# Route trang chính
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Route lấy dữ liệu cảm biến
@app.route('/sensor_data')
@login_required
def sensor_data():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    feeds = ['temperature', 'humidity', 'soilmoisture']
    data = {}
    try:
        for feed in feeds:
            url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/{feed}/data/last"
            response = requests.get(url, headers=AIO_HEADERS)
            response.raise_for_status()
            value = response.json()['value']
            data[feed] = value
        pump_state = requests.get(f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/relaycontrol/data/last", headers=AIO_HEADERS).json()['value']
        new_sensor_data = SensorData(
            user_id=user.id,
            temperature=float(data['temperature']) if data['temperature'] else None,
            humidity=float(data['humidity']) if data['humidity'] else None,
            soilmoisture=float(data['soilmoisture']) if data['soilmoisture'] else None,
            pump_state=pump_state
        )
        db.session.add(new_sensor_data)
        db.session.commit()
        logging.debug(f"Dữ liệu cảm biến: {data}")
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi lấy dữ liệu cảm biến: {str(e)}")
        return jsonify({'error': f'Không lấy được dữ liệu từ Adafruit IO: {str(e)}'})

# Route điều khiển
@app.route('/control', methods=['POST'])
@login_required
def control():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    feed = request.form['feed']
    value = request.form['value']
    success = send_feed_data(feed, value)
    if success and feed in ['relaycontrol', 'autoenable']:
        config = Configuration.query.filter_by(user_id=user.id).first()
        if feed == 'relaycontrol':
            latest_sensor = SensorData.query.filter_by(user_id=user.id).order_by(SensorData.timestamp.desc()).first()
            if latest_sensor:
                latest_sensor.pump_state = value
                db.session.commit()
        elif feed == 'autoenable':
            config.auto_mode = value
            config.last_updated = datetime.utcnow()
            db.session.commit()
            # Kiểm tra ngay lập tức khi bật chế độ tự động
            if value == 'ON':
                try:
                    # Lấy dữ liệu độ ẩm đất
                    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/soilmoisture/data/last"
                    response = requests.get(url, headers=AIO_HEADERS)
                    response.raise_for_status()
                    soil_moisture = float(response.json()['value'])
                    logging.debug(f"Độ ẩm đất của user {user.id} khi bật auto: {soil_moisture}%")
                except (requests.exceptions.RequestException, ValueError) as e:
                    logging.error(f"Lỗi lấy dữ liệu độ ẩm đất ngay lập tức cho user {user.id}: {str(e)}")
                    return jsonify({'success': True})

                # Lấy trạng thái bơm hiện tại
                try:
                    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/relaycontrol/data/last"
                    response = requests.get(url, headers=AIO_HEADERS)
                    response.raise_for_status()
                    pump_state = response.json()['value']
                    logging.debug(f"Trạng thái bơm của user {user.id} khi bật auto: {pump_state}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Lỗi lấy trạng thái bơm ngay lập tức cho user {user.id}: {str(e)}")
                    pump_state = 'OFF'

                # Kiểm tra và điều khiển relay ngay lập tức
                if soil_moisture < config.soil_min and pump_state != 'ON':
                    success = send_feed_data('relaycontrol', 'ON')
                    if success:
                        logging.info(f"Độ ẩm đất ({soil_moisture}%) thấp hơn mục tiêu ({config.soil_min}%), bật bơm ngay lập tức cho user {user.id}")
                        latest_sensor = SensorData.query.filter_by(user_id=user.id).order_by(SensorData.timestamp.desc()).first()
                        if latest_sensor:
                            latest_sensor.pump_state = 'ON'
                            db.session.commit()
                    else:
                        logging.error(f"Thất bại khi bật bơm ngay lập tức cho user {user.id}")
                elif soil_moisture >= config.soil_max and pump_state != 'OFF':
                    success = send_feed_data('relaycontrol', 'OFF')
                    if success:
                        logging.info(f"Độ ẩm đất ({soil_moisture}%) vượt mục tiêu ({config.soil_max}%), tắt bơm ngay lập tức cho user {user.id}")
                        latest_sensor = SensorData.query.filter_by(user_id=user.id).order_by(SensorData.timestamp.desc()).first()
                        if latest_sensor:
                            latest_sensor.pump_state = 'OFF'
                            db.session.commit()
                    else:
                        logging.error(f"Thất bại khi tắt bơm ngay lập tức cho user {user.id}")
                else:
                    logging.debug(f"Không cần điều chỉnh bơm ngay lập tức cho user {user.id}: soil_moisture={soil_moisture}%, soil_min={config.soil_min}%, soil_max={config.soil_max}%, pump_state={pump_state}")
    logging.info(f"Control request: feed={feed}, value={value}, success={success}")
    return jsonify({'success': True})

# Route lấy trạng thái bơm
@app.route('/pump_state')
@login_required
def pump_state():
    try:
        url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/relaycontrol/data/last"
        response = requests.get(url, headers=AIO_HEADERS)
        response.raise_for_status()
        state = response.json()['value']
        logging.debug(f"Trạng thái bơm từ Adafruit IO: {state}")
        return jsonify({'state': state})
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi lấy trạng thái bơm: {str(e)}")
        return jsonify({'state': 'OFF', 'error': str(e)})

# Route lấy trạng thái chế độ tự động
@app.route('/auto_state')
@login_required
def auto_state():
    try:
        url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/autoenable/data/last"
        response = requests.get(url, headers=AIO_HEADERS)
        response.raise_for_status()
        state = response.json()['value']
        logging.debug(f"Trạng thái chế độ tự động từ Adafruit IO: {state}")
        return jsonify({'state': state})
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi lấy trạng thái chế độ tự động: {str(e)}")
        return jsonify({'state': 'OFF', 'error': str(e)})

# Route cập nhật mục tiêu độ ẩm đất
@app.route('/update_soil_target', methods=['POST'])
@login_required
def update_soil_target():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    soil_min = request.form.get('soil_min')
    soil_max = request.form.get('soil_max')
    if not soil_min or not soil_max or float(soil_min) > float(soil_max):
        logging.error("Giá trị mục tiêu độ ẩm đất không hợp lệ")
        return jsonify({'error': 'Giá trị mục tiêu không hợp lệ'}), 400
    config = Configuration.query.filter_by(user_id=user.id).first()
    config.soil_min = float(soil_min)
    config.soil_max = float(soil_max)
    config.last_updated = datetime.utcnow()
    db.session.commit()
    send_feed_data('soilmoisturetarget', f"{soil_min}-{soil_max}")
    return jsonify({'success': True})

# Route lấy dữ liệu lịch sử cảm biến
@app.route('/sensor_history', methods=['GET'])
@login_required
def sensor_history():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = SensorData.query.filter_by(user_id=user.id)
    if start_date:
        query = query.filter(SensorData.timestamp >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.filter(SensorData.timestamp <= datetime.fromisoformat(end_date))
    
    pagination = query.order_by(SensorData.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    sensor_data = pagination.items
    total = pagination.total
    
    return jsonify({
        'data': [{
            'timestamp': data.timestamp.isoformat(),
            'temperature': data.temperature,
            'humidity': data.humidity,
            'soilmoisture': data.soilmoisture,
            'pump_state': data.pump_state
        } for data in sensor_data],
        'total': total
    })

# Route dự đoán
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    if model is None:
        logging.error("Mô hình học máy không khả dụng trong predict")
        return jsonify({'error': 'Mô hình học máy không khả dụng. Vui lòng kiểm tra ensemble_model.pkl'})
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        input_data = pd.DataFrame([data])
        input_data["SoilType"] = encoder_soil.transform(input_data["SoilType"])
        input_data["CropType"] = encoder_crop.transform(input_data["CropType"])
        input_data[["Temparature", "Humidity", "Moisture"]] = \
            scaler.transform(input_data[["Temparature", "Humidity", "Moisture"]])
        input_water = input_data[feature_columns]
        predicted_water, all_model_outputs = model.predict_best(input_water)
        # Ép kiểu predicted_water thành float
        predicted_water = float(predicted_water)
        # Ép kiểu tất cả các giá trị trong all_model_outputs thành float
        all_model_outputs = {model: float(value) for model, value in all_model_outputs.items()}
        logging.info(f"Predicted Water Usage: {predicted_water} l")
        logging.info(f"All Model Outputs: {all_model_outputs}")
        send_feed_data('water', str(predicted_water))
        response_data = {
            "PredictedWaterUsage": predicted_water,
            "AllModelOutputs": all_model_outputs
        }
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Route lập lịch
@app.route('/schedule', methods=['POST'])
@login_required
def schedule():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    try:
        scheduled_time_str = request.form['scheduled_time']
        off_time_str = request.form['off_time']
        repeat_daily = request.form.get('repeat_daily') == 'on'
        scheduled_time = datetime.strptime(scheduled_time_str, '%Y-%m-%dT%H:%M')
        off_time = datetime.strptime(off_time_str, '%Y-%m-%dT%H:%M')
        if off_time <= scheduled_time:
            logging.error("Thời gian tắt phải sau thời gian bật")
            return jsonify({'error': 'Thời gian tắt phải sau thời gian bật'})
        new_schedule = Schedule(
            user_id=user.id,
            scheduled_time=scheduled_time,
            off_time=off_time,
            executed=False,
            repeat_daily=repeat_daily
        )
        db.session.add(new_schedule)
        db.session.commit()
        logging.info(f"Đã lập lịch: Bật lúc {scheduled_time}, Tắt lúc {off_time}, Lặp lại: {repeat_daily}")
        return jsonify({'success': True})
    except ValueError as e:
        logging.error(f"Lỗi dữ liệu lịch: {str(e)}")
        return jsonify({'error': 'Dữ liệu lịch không hợp lệ'})
    except Exception as e:
        logging.error(f"Lỗi trong schedule: {str(e)}")
        return jsonify({'error': str(e)})

# Route lấy danh sách lịch
@app.route('/schedules')
@login_required
def get_schedules():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    schedules = Schedule.query.filter_by(user_id=user.id, executed=False).all()
    logging.debug(f"Lấy danh sách lịch: {len(schedules)} mục")
    return jsonify([{
        'id': s.id,
        'scheduled_time': s.scheduled_time.isoformat(),
        'off_time': s.off_time.isoformat(),
        'repeat_daily': s.repeat_daily
    } for s in schedules])

# Route xóa lịch
@app.route('/delete_schedule/<int:id>', methods=['DELETE'])
@login_required
def delete_schedule(id):
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    try:
        schedule = Schedule.query.filter_by(id=id, user_id=user.id).first()
        if not schedule:
            logging.error(f"Không tìm thấy lịch với ID {id}")
            return jsonify({'error': 'Không tìm thấy lịch'}), 404
        db.session.delete(schedule)
        db.session.commit()
        logging.info(f"Đã xóa lịch với ID {id}")
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Lỗi khi xóa lịch: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route kiểm tra kết nối
@app.route('/check_connection')
@login_required
def check_connection():
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    try:
        url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/temperature/data/last"
        response = requests.get(url, headers=AIO_HEADERS)
        response.raise_for_status()
        value = response.json()['value']
        logging.info(f"Kiểm tra kết nối thành công: {value}")
        return jsonify({'status': 'connected', 'message': f'Đã kết nối, giá trị mẫu: {value}'})
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi kiểm tra kết nối: {str(e)}")
        return jsonify({'status': 'disconnected', 'message': str(e)})

# Route trang admin
@app.route('/admin')
@admin_required
def admin_dashboard():
    try:
        return render_template('admin.html')
    except Exception as e:
        logging.error(f"Lỗi khi tải trang admin: {str(e)}")
        flash(f'Lỗi khi tải trang admin: {str(e)}', 'error')
        return redirect(url_for('index'))

# Route lấy danh sách người dùng
@app.route('/admin/users')
@admin_required
def get_users():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        query = request.args.get('query', '').lower()
        
        users_query = User.query
        if query:
            users_query = users_query.filter(User.username.ilike(f'%{query}%'))
        
        pagination = users_query.paginate(page=page, per_page=per_page, error_out=False)
        users = pagination.items
        total = pagination.total
        
        return jsonify({
            'users': [{
                'id': user.id,
                'username': user.username,
                'role': user.role
            } for user in users],
            'total': total
        })
    except Exception as e:
        logging.error(f"Lỗi khi lấy danh sách người dùng: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route lấy dữ liệu cảm biến của một người dùng (cho admin)
@app.route('/admin/sensor_data/<int:user_id>')
@admin_required
def admin_sensor_data(user_id):
    user = User.query.filter_by(username=session['username']).first()
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = SensorData.query.filter_by(user_id=user_id)
        if start_date:
            query = query.filter(SensorData.timestamp >= datetime.fromisoformat(start_date))
        if end_date:
            query = query.filter(SensorData.timestamp <= datetime.fromisoformat(end_date))
        
        pagination = query.order_by(SensorData.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        sensor_data = pagination.items
        total = pagination.total
        
        return jsonify({
            'data': [{
                'timestamp': data.timestamp.isoformat(),
                'temperature': data.temperature,
                'humidity': data.humidity,
                'soilmoisture': data.soilmoisture,
                'pump_state': data.pump_state
            } for data in sensor_data],
            'total': total
        })
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu cảm biến cho user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route quản lý người dùng (admin)
@app.route('/admin/user/<int:user_id>', methods=['GET', 'POST', 'DELETE'])
@admin_required
def manage_user(user_id):
    user = User.query.get_or_404(user_id)
    if user is None:
        logging.error("Người dùng không tồn tại trong cơ sở dữ liệu")
        return jsonify({'error': 'Người dùng không tồn tại'}), 401
    if request.method == 'POST':
        username = request.form.get('username')
        role = request.form.get('role')
        if username and role in ['user', 'admin']:
            user.username = username
            user.role = role
            db.session.commit()
            logging.info(f"Cập nhật user {user_id} thành công: username={username}, role={role}")
            return jsonify({'success': True})
        logging.error("Dữ liệu không hợp lệ khi cập nhật user")
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400
    elif request.method == 'DELETE':
        db.session.delete(user)
        db.session.commit()
        logging.info(f"Xóa user {user_id} thành công")
        return jsonify({'success': True})
    return jsonify({
        'id': user.id,
        'username': user.username,
        'role': user.role
    })

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            logging.info("Khởi tạo cơ sở dữ liệu thành công")
        except Exception as e:
            logging.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {str(e)}")
    threading.Thread(target=check_schedules, daemon=True).start()
    threading.Thread(target=check_auto_mode, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)