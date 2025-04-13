from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from io import BytesIO
import boto3
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go
from datetime import datetime
import base64
from PIL import Image
import pytesseract
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')  # Can be removed if not needed

# Load YOLO model
try:
    model = YOLO('model/best.pt')
    logger.info("YOLO model loaded successfully")
    logger.info(f"Model names: {model.names}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

# Dummy cost estimator with reference table (USD adjusted for India)
class CostEstimator:
    damage_costs = {
        'Front fender': {'Slight Scratch': 205, 'Severe Scratch': 340, 'Slight Dent': 340, 'Severe Dent': 805, 'Replacement': 805},
        'Front bonnet': {'Slight Scratch': 205, 'Severe Scratch': 405, 'Slight Dent': 375, 'Severe Dent': 935, 'Replacement': 935},
        'Rear door': {'Slight Scratch': 140, 'Severe Scratch': 270, 'Slight Dent': 270, 'Severe Dent': 470, 'Replacement': 805},
        'Roof': {'Slight Scratch': 340, 'Severe Scratch': 470, 'Slight Dent': 405, 'Severe Dent': 670, 'Replacement': 1940},
        'Rear trunk': {'Slight Scratch': 205, 'Severe Scratch': 375, 'Slight Dent': 375, 'Severe Dent': 600, 'Replacement': 1455},
        'Front bumper': {'Slight Scratch': 125, 'Severe Scratch': 245, 'Slight Dent': 245, 'Severe Dent': 675, 'Replacement': 675},
        'Windshield': {'Slight Scratch': 630, 'Severe Scratch': 630, 'Slight Dent': 630, 'Severe Dent': 630, 'Replacement': 630},
        'full frontal damage': {'Slight Damage': 170, 'Severe Damage': 14110}
    }
    INDIA_ADJUSTMENT_FACTOR = 0.5

    def estimate_detailed_cost(self, damage_type, confidence, image_area, damage_area, vehicle_category, vehicle_info):
        default_part = 'Front fender'
        if 'dent' in damage_type.lower():
            severity = 'Severe' if confidence > 0.7 else 'Slight'
            damage_category = f'{severity} Dent'
        elif 'scratch' in damage_type.lower():
            severity = 'Severe' if confidence > 0.7 else 'Slight'
            damage_category = f'{severity} Scratch'
        elif 'full frontal damage' in damage_type.lower():
            damage_category = 'Severe Damage' if confidence > 0.7 else 'Slight Damage'
        else:
            damage_category = 'Slight Dent'

        part_costs = self.damage_costs.get(damage_type, self.damage_costs[default_part])
        cost_usd = part_costs.get(damage_category, part_costs.get('Slight Dent', 340))
        adjusted_cost_usd = cost_usd * self.INDIA_ADJUSTMENT_FACTOR
        cost_inr = adjusted_cost_usd * 83
        return {'total': cost_inr, 'parts': cost_inr * 0.6, 'labor': cost_inr * 0.4}

cost_estimator = CostEstimator()

# Number plate detection functions
def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    return 1063.62 <= area <= 73862.5 and 3 <= ratio <= 6

def isMaxWhite(plate):
    avg = np.mean(plate)
    return avg >= 115

def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect
    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle
    if angle > 15 or height == 0 or width == 0:
        return False
    area = height * width
    return ratioCheck(area, width, height)

def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)
        max_cnt = num_contours[max_cntr_index]
        max_cntArea = contour_area[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        if not ratioCheck(max_cntArea, w, h):
            return plate, None
        
        final_img = thresh[y:y+h, x:x+w]
        return final_img, [x, y, w, h]
    return plate, None

def number_plate_detection(img):
    try:
        logger.info("Starting number plate detection")
        img2 = cv2.GaussianBlur(img, (5, 5), 0)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
        morph_img_threshold = img2.copy()
        cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        num_contours, _ = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        
        for cnt in num_contours:
            min_rect = cv2.minAreaRect(cnt)
            if ratio_and_rotation(min_rect):
                x, y, w, h = cv2.boundingRect(cnt)
                plate_img = img[y:y+h, x:x+w]
                logger.info(f"Found potential plate region: x={x}, y={y}, w={w}, h={h}")
                if isMaxWhite(plate_img):
                    clean_plate, rect = clean2_plate(plate_img)
                    if rect:
                        x1, y1, w1, h1 = rect
                        x, y, w, h = x + x1, y + y1, w1, h1
                        plate_im = Image.fromarray(clean_plate)
                        for psm in [6, 8, 10]:
                            text = pytesseract.image_to_string(plate_im, lang='eng', config=f'--psm {psm}')
                            cleaned_text = str("".join(re.split("[^a-zA-Z0-9]*", text))).upper()
                            logger.info(f"PSM {psm} OCR result: {cleaned_text}")
                            if cleaned_text and len(cleaned_text) >= 4:
                                return cleaned_text
        logger.warning("No valid number plate detected")
        return None
    except Exception as e:
        logger.error(f"Number plate detection error: {str(e)}")
        return None

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/assess-damage', methods=['POST'])
def assess_damage():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    vehicle_category = request.form.get('vehicle_category', 'mid_range')
    vehicle_make = request.form.get('vehicle_make', '')
    vehicle_model = request.form.get('vehicle_model', '')
    vehicle_year = request.form.get('vehicle_year', '2023')
    vehicle_info = {'make': vehicle_make, 'model': vehicle_model, 'year': vehicle_year}

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    result_image, detections = process_image(file, vehicle_category, vehicle_info)
    if result_image is None:
        return jsonify({'error': 'Failed to process image'}), 500

    report = generate_report(detections, vehicle_info, cost_estimator, vehicle_category)
    image_data = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode('utf-8')
    visualization = generate_visualization(report['cost_breakdown'])

    return jsonify({
        'image_data': image_data,
        'report': report,
        'visualization': visualization
    })

@app.route('/api/assess-video', methods=['POST'])
def assess_video():
    if 'image' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    vehicle_category = request.form.get('vehicle_category', 'mid_range')
    vehicle_make = request.form.get('vehicle_make', '')
    vehicle_model = request.form.get('vehicle_model', '')
    vehicle_year = request.form.get('vehicle_year', '2023')
    vehicle_info = {'make': vehicle_make, 'model': vehicle_model, 'year': vehicle_year}

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    result_image, detections = process_video(file, vehicle_category, vehicle_info)
    if result_image is None:
        return jsonify({'error': 'Failed to process video'}), 500

    report = generate_report(detections, vehicle_info, cost_estimator, vehicle_category)
    image_data = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode('utf-8') if result_image is not None else ''
    visualization = generate_visualization(report['cost_breakdown'])

    return jsonify({
        'image_data': image_data,
        'report': report,
        'visualization': visualization
    })

@app.route('/api/detect-number-plate', methods=['POST'])
def detect_number_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 500

    number_plate = number_plate_detection(img)
    if number_plate:
        return jsonify({'number_plate': number_plate})
    else:
        return jsonify({'number_plate': None, 'message': 'No number plate detected'})

@app.route('/api/webcam', methods=['POST'])
def webcam():
    vehicle_category = request.form.get('vehicle_category', 'mid_range')
    vehicle_make = request.form.get('vehicle_make', '')
    vehicle_model = request.get('vehicle_model', '')
    vehicle_year = request.form.get('vehicle_year', '2023')
    vehicle_info = {'make': vehicle_make, 'model': vehicle_model, 'year': vehicle_year}

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    result_image, detections = process_webcam(vehicle_category, vehicle_info)
    if result_image is None:
        return jsonify({'error': 'Failed to process webcam'}), 500

    report = generate_report(detections, vehicle_info, cost_estimator, vehicle_category)
    image_data = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode('utf-8')
    visualization = generate_visualization(report['cost_breakdown'])

    return jsonify({
        'image_data': image_data,
        'report': report,
        'visualization': visualization
    })

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        if not data or 'report' not in data:
            logger.error("No report data provided in request")
            return jsonify({'error': 'No report data provided'}), 400
        report = data['report']
        logger.info(f"Generating PDF with report: {report}")

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(f"Damage Assessment Report - {timestamp}", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Vehicle: {report['vehicle_info']['make']} {report['vehicle_info']['model']} ({report['vehicle_info'].get('year', 'N/A')})", styles['Normal']))
        story.append(Paragraph(f"Number Plate: {report.get('number_plate', 'Not detected')}", styles['Normal']))
        story.append(Paragraph(f"Total Cost: ₹{report['cost_breakdown']['total']:.2f}", styles['Normal']))
        story.append(Paragraph(f"Estimated Repair Time: {report['repair_time_estimate']} days", styles['Normal']))
        story.append(Paragraph("Damage Summary:", styles['Heading2']))
        for damage in report.get('damage_summary', []):
            story.append(Paragraph(f"- {damage['damage_type']}: {damage['severity']} (Confidence: {damage['confidence']*100:.2f}%)", styles['Normal']))
        doc.build(story)
        buffer.seek(0)
        logger.info("PDF generated successfully")
        return send_file(buffer, as_attachment=True, download_name=f'damage_assessment_report_{timestamp.replace(":", "_")}.pdf', mimetype='application/pdf')
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

# Processing functions
def process_image(file, vehicle_category, vehicle_info):
    try:
        logger.info(f"Processing image: {file.filename}")
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Invalid image file")
            return None, []
        results = model(image)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                damage_type = model.names[cls]
                if conf > 0.5:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, f"{damage_type} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    detections.append({'damage_type': damage_type, 'confidence': conf, 'severity': 'moderate'})
        return image, detections
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None, []

def process_video(file, vehicle_category, vehicle_info):
    try:
        logger.info(f"Processing video: {file.filename}")
        temp_path = 'temp_video.mp4'
        file.save(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("Unable to open video file")
            os.remove(temp_path)
            return None, []

        detections = []
        frame_count = 0
        max_frames = 100
        frame_step = 1

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_step == 0:
                logger.info(f"Processing frame {frame_count}")
                results = model(frame)
                logger.info(f"Frame {frame_count} detections: {len(results[0].boxes)}")
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        damage_type = model.names[cls]
                        if conf > 0.5:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{damage_type} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            detections.append({'damage_type': damage_type, 'confidence': conf, 'severity': 'moderate'})
            
            frame_count += 1

        cap.release()
        os.remove(temp_path)

        if detections:
            logger.info(f"Total detections: {len(detections)}")
            return frame, detections
        else:
            logger.warning("No damage detected in video, returning fallback")
            return frame, []

    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        if 'temp_path' in locals():
            os.remove(temp_path)
        return None, []

def process_webcam(vehicle_category, vehicle_info):
    try:
        logger.info("Processing webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Webcam not accessible")
            return None, []
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture webcam frame")
            cap.release()
            return None, []
        cap.release()
        results = model(frame)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                damage_type = model.names[cls]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{damage_type} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detections.append({'damage_type': damage_type, 'confidence': conf, 'severity': 'moderate'})
        return frame, detections
    except Exception as e:
        logger.error(f"Webcam processing error: {str(e)}")
        return None, []

def generate_report(detections, vehicle_info, cost_estimator, vehicle_category, number_plate=None, user_damage_cost=None, repair_days=None):
    if not detections and not user_damage_cost:
        return {'cost_breakdown': {'total': 0}, 'repair_time_estimate': 'N/A', 'damage_summary': [], 'vehicle_info': vehicle_info, 'number_plate': number_plate}
    total_cost = 0
    damage_summary = []
    for det in detections:
        cost = cost_estimator.estimate_detailed_cost(det['damage_type'], det['confidence'], 1000, 100, vehicle_category, vehicle_info)
        total_cost += cost['total']
        damage_summary.append({'damage_type': det['damage_type'], 'severity': det['severity'], 'confidence': det['confidence']})
    # Override with user-provided cost if available
    if user_damage_cost is not None:
        total_cost = float(user_damage_cost)
    return {
        'cost_breakdown': {'total': total_cost, 'parts': total_cost * 0.6, 'labor': total_cost * 0.4},
        'repair_time_estimate': repair_days if repair_days else len(detections) * 2,
        'damage_summary': damage_summary,
        'vehicle_info': vehicle_info,
        'number_plate': number_plate,
        'user_damage_cost': user_damage_cost
    }

def generate_visualization(cost_breakdown):
    fig = go.Figure(data=[go.Pie(labels=['Parts', 'Labor'], values=[cost_breakdown.get('parts', 0), cost_breakdown.get('labor', 0)])])
    fig.update_layout(width=400, height=300)
    return fig.to_html(include_plotlyjs='cdn', full_html=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)