from flask import Flask, render_template, request, redirect, session, send_file
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from PIL import Image
import cv2
import numpy as np
from skimage import color
from utils.wrist_detection import predict_wrist
from utils.generate_products_code import generate_product_code

app = Flask(__name__)
app.secret_key = 'rahasia'
app.config['UPLOAD_PRODUCTS'] = 'static/uploads/products'       # digunakan untuk menyimpan gambar makeup
app.config['UPLOAD_RESULTS'] = 'static/uploads/results'         # digunakan untuk menyimpan sementara gambar hasil prediksi
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

admin_user = {'username': 'admin', 'password': 'admin123'}

class Makeup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    products_code = db.Column(db.String(36), unique=True, default=lambda: str(generate_product_code()), nullable=False) # membuat product kode produts agar mudah dibaca
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    skin_tone = db.Column(db.String(50), nullable=False)  # Gantilah tone dengan skin_tone
    undertone = db.Column(db.String(50), nullable=False)  # Menambahkan undertone
    image = db.Column(db.String(100), nullable=False)

class Knowledge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    skin_tone = db.Column(db.String(100))
    undertone = db.Column(db.String(100))
    recommendation = db.Column(db.Text)
    description = db.Column(db.String(100))

def preprocess_image(image_path):
    
    image = predict_wrist(image_path)       # preprosessing dilakukan oleh dengan memprediksi bouding box (dilakukan oleh machine learning)
                                            # hasil preprossing img ditampilkan dalam pdf (hasil prediksi)
    if image is None or image.size == 0:
        return None
    
    resized = cv2.resize(image, (100, 100))
    return resized

def detect_dominant_color(image, k=3):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    b, g, r = dominant.astype(int)
    return r, g, b

REFERENCE_SKIN = {
    "light_warm":    (242, 210, 170),  # Putih gading, undertone kekuningan
    "light_neutral": (235, 200, 180),  # Putih terang, undertone seimbang
    "medium_warm":   (205, 160, 120),  # Sawo matang cerah, warm undertone
    "medium_neutral":(190, 150, 130),  # Sawo matang netral
    "medium_cool":   (180, 145, 160),  # Sawo matang kemerahan (jarang)
    "dark_warm":     (135, 100, 70),   # Gelap kekuningan, seperti warna kulit Papua
    "dark_neutral":  (120, 90, 75),    # Gelap netral, sering ditemukan di bagian Timur Indonesia
}

def classify_by_rgb_distance(r, g, b):
    min_dist = float('inf')
    label = "neutral"
    for key, val in REFERENCE_SKIN.items():
        dist = np.sqrt((r - val[0])**2 + (g - val[1])**2 + (b - val[2])**2)
        if dist < min_dist:
            min_dist = dist
            label = key
    parts = label.split('_')
    skin_tone = parts[0]
    undertone = parts[1] if len(parts) > 1 else "neutral"
    return skin_tone, undertone

def get_rules_from_db():
    knowledge_list = Knowledge.query.all()
    rules = []
    for item in knowledge_list:
        rules.append({
            "conditions": {
                "skin_tone": item.skin_tone.lower(),
                "undertone": item.undertone.lower()
            },
            "recommendations": [line.strip() for line in item.recommendation.split(',') if line.strip()]
        })

    return rules

# Forward chaining algorithm to find recommendations
def forward_chaining(skin_tone, undertone, rules):
    recommendations = []
    for rule in rules:
        if rule["conditions"]["skin_tone"] == skin_tone and rule["conditions"]["undertone"] == undertone:
            for recommendation in rule["recommendations"]:
                recommendations.append(recommendation)

    makeup_items = Makeup.query.filter(Makeup.products_code.in_(recommendations)).all()    # Ambil produk yang relevan
    return makeup_items

user_result = {
    "skin_tone": "",
    "undertone": "",
    "recommendations": []
}

errorDetectingWrist = None
cropped_img_array = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global user_result
    global errorDetectingWrist
    global cropped_img_array
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_RESULTS'], filename)
            os.makedirs(app.config['UPLOAD_RESULTS'], exist_ok=True)
            file.save(filepath)

            # Deteksi warna dominan
            image = preprocess_image(filepath)

            if os.path.exists(filepath):
                os.remove(filepath)

            if image is None or image.size == 0:
                errorDetectingWrist = True
                return redirect('/result')
            else:
                errorDetectingWrist = False

            cropped_img_array = image
            r, g, b = detect_dominant_color(image)
            skin_tone, undertone = classify_by_rgb_distance(r, g, b)

            # Ambil pengetahuan dari Knowledge
            rules = get_rules_from_db()
            recommendations = forward_chaining(skin_tone, undertone, rules)

            user_result = {
                "skin_tone": skin_tone,
                "undertone": undertone,
                "recommendations": recommendations  # Hanya rekomendasi berdasarkan pengetahuan
            }

            return redirect('/result')

    return render_template('home.html')


@app.route('/result')
def result():
    if (errorDetectingWrist == None):
        return redirect("/")
    
    return render_template('result.html', result=user_result, errorOccur=errorDetectingWrist)

@app.route('/download')
def download():
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Hasil Prediksi", ln=True, align='C')
        pdf.ln(8)

        img_path = "static/uploads/results/temp_image.jpg"
        cv2.imwrite(img_path, cropped_img_array)

        # Dapatkan lebar gambar dalam mm untuk PDF
        img_width = 50  # bisa disesuaikan
        x_center = (pdf.w - img_width) / 2

        # Tambahkan gambar ke PDF
        pdf.image(img_path, x=x_center, w=img_width)
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Skin Tone: {user_result.get('skin_tone', '')}", ln=True)
        pdf.cell(0, 10, f"Undertone: {user_result.get('undertone', '')}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(0, 10, "Rekomendasi Produk Makeup", ln=True)

        pdf.set_font("Arial", size=13)
        for item in user_result.get("recommendations")[:10]:
            pdf.multi_cell(0, 10, f"- {item.name} ({item.category})") # akan lanjut ke baris baru jika rekomendasi terlalu panjang


        pdf.output("static/hasil.pdf")

        if os.path.exists(img_path): # menghapus file gambar setelah selesai
            os.remove(img_path)

        return send_file("static/hasil.pdf", as_attachment=True)
    
    except:
        return redirect("/")


@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == admin_user['username'] and password == admin_user['password']:
            session['admin'] = True
            return redirect('/admin')
    return render_template('login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect('/')

@app.route('/admin')
def admin_dashboard():
    if not session.get('admin'):
        return redirect('/admin/login')
    makeup = Makeup.query.all()
    return render_template('admin_dashboard.html', makeup=makeup)

@app.route('/admin/add', methods=['GET', 'POST'])
def add_makeup():
    if not session.get('admin'):
        return redirect('/admin/login')
    if request.method == 'POST':
        name = request.form.get('name')
        category = request.form.get('category')
        skin_tone = request.form.get('skin_tone')
        undertone = request.form.get('undertone')
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_PRODUCTS'], filename)
            os.makedirs(app.config['UPLOAD_PRODUCTS'], exist_ok=True)
            file.save(filepath)
            print(name, category, skin_tone, undertone, file)
            new_item = Makeup(name=name, category=category, skin_tone=skin_tone, undertone=undertone, image=filename)
            # db.session.add(new_item)
            # db.session.commit()

            try:
                db.session.add(new_item)
                db.session.commit()
                print("Data berhasil disimpan")
            except Exception as e:
                print("Gagal simpan ke DB:", e)

            return redirect('/admin')
    return render_template('form_add_edit.html', form_title='Tambah Produk', item=None)

@app.route('/admin/edit/<int:id>', methods=['GET', 'POST'])
def edit_makeup(id):
    if not session.get('admin'):
        return redirect('/admin/login')
    item = Makeup.query.get_or_404(id)
    if request.method == 'POST':
        item.name = request.form.get('name')
        item.category = request.form.get('category')
        item.skin_tone = request.form.get('skin_tone')
        item.undertone = request.form.get('undertone')
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_PRODUCTS'], filename)
            os.makedirs(app.config['UPLOAD_PRODUCTS'], exist_ok=True)
            file.save(filepath)
            item.image = filename
        db.session.commit()
        return redirect('/admin')
    return render_template('form_add_edit.html', form_title='Edit Produk', item=item)

@app.route('/admin/delete/<int:id>')
def delete_makeup(id):
    if not session.get('admin'):
        return redirect('/admin/login')
    product = Makeup.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    return redirect('/admin')

@app.route('/admin/knowledge')
def admin_knowledge():
    if not session.get('admin'):
        return redirect('/admin/login')
    knowledge = Knowledge.query.all()
    return render_template('admin_knowledge.html', knowledge=knowledge)

@app.route('/admin/knowledge/add', methods=['GET', 'POST'])
def add_knowledge():
    if not session.get('admin'):
        return redirect('/admin/login')
    if request.method == 'POST':
        skin_tone = request.form.get('skin_tone')
        undertone = request.form.get('undertone')
        recommendation = request.form.get('recommendation')
        description = request.form.get('description')
        new_knowledge = Knowledge(skin_tone=skin_tone, undertone=undertone,
                                   recommendation=recommendation, description=description)
        db.session.add(new_knowledge)
        db.session.commit()
        return redirect('/admin/knowledge')
    return render_template('form_add_knowledge.html', item=None)

@app.route('/admin/knowledge/edit/<int:id>', methods=['GET', 'POST'])
def edit_knowledge(id):
    if not session.get('admin'):
        return redirect('/admin/login')
    item = Knowledge.query.get_or_404(id)
    if request.method == 'POST':
        item.skin_tone = request.form.get('skin_tone')
        item.undertone = request.form.get('undertone')
        item.recommendation = request.form.get('recommendation')
        item.description = request.form.get('description')
        db.session.commit()
        return redirect('/admin/knowledge')
    return render_template('form_add_knowledge.html', item=item)

@app.route('/admin/knowledge/delete/<int:id>')
def delete_knowledge(id):
    if not session.get('admin'):
        return redirect('/admin/login')
    item = Knowledge.query.get_or_404(id)
    db.session.delete(item)
    db.session.commit()
    return redirect('/admin/knowledge')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    with app.app_context():
        # db.drop_all()
        db.create_all() 
    app.run(debug=True)
