import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import pandas as pd
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)

# Configuración de Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Cargar modelo al iniciar
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'modelo', 'impresion.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    # Optimizar modelo
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    
    return model

model = load_model()

def send_telegram_alert(image, detections):
    """Envía una alerta a Telegram con la imagen y las detecciones (excepto 'imprimiendo')"""
    try:
        # Convertir detecciones a DataFrame si no lo son
        if not isinstance(detections, pd.DataFrame):
            detections = pd.DataFrame(detections)
        
        # Filtrar detecciones para excluir 'imprimiendo'
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        
        # Si no hay detecciones después del filtro, no enviar alerta
        if filtered_detections.empty:
            print("No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return False
        
        # Convertir la imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            print("Error al codificar la imagen")
            return False

        # Enviar la foto
        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        # Crear mensaje con las detecciones filtradas
        message = "⚠ *Detección de error en impresión 3D* ⚠\n\n"
        for _, row in filtered_detections.iterrows():
            message += f"🔹 *{row['name']}*\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        # Primero enviar la foto con el caption
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)

        if response.status_code != 200:
            print(f"Error al enviar alerta a Telegram: {response.text}")
            return False
        else:
            print("Alerta enviada a Telegram (errores detectados)")
            return True
            
    except Exception as e:
        print(f"Error en send_telegram_alert: {str(e)}")
        return False

@app.route('/detect', methods=['POST'])
def detect_errors():
    try:
        # Verificar si se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Leer imagen
        file = request.files['image']
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Realizar detección
        results = model(img)
        detections = results.pandas().xyxy[0]
        
        # Renderizar imagen con detecciones
        result_img = np.squeeze(results.render())
        
        # Verificar si hay errores y enviar alerta
        has_errors = not detections[detections['name'].str.lower() != 'imprimiendo'].empty
        alert_sent = False
        
        if has_errors:
            alert_sent = send_telegram_alert(result_img, detections)
        
        # Preparar respuesta
        response = {
            "detections": detections.to_dict(orient='records'),
            "has_errors": has_errors,
            "alert_sent": alert_sent
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)