import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import requests
import traceback

# Usar Ultralytics en lugar de YOLOv5 de torch.hub
from ultralytics import YOLO

app = Flask(__name__)

# Configuración de Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Variable global para el modelo
model = None

def load_model():
    """Carga el modelo usando Ultralytics YOLO"""
    try:
        print("🔄 Cargando modelo con Ultralytics...")
        
        # Verificar que el archivo del modelo existe
        model_path = os.path.join(os.path.dirname(__file__), 'modelo', 'impresion.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")
        
        print(f"📁 Cargando modelo desde: {model_path}")
        
        # Cargar modelo con Ultralytics (más estable)
        model = YOLO(model_path)
        
        # Configurar parámetros
        model.overrides['conf'] = 0.25
        model.overrides['iou'] = 0.45
        model.overrides['agnostic_nms'] = False
        model.overrides['max_det'] = 1000
        
        print("✅ Modelo cargado exitosamente con Ultralytics")
        return model
        
    except Exception as e:
        print(f"❌ Error cargando el modelo: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def process_ultralytics_results(results):
    """Convierte resultados de Ultralytics a formato pandas"""
    detections_list = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                detection = {
                    'xmin': float(box.xyxy[0][0]),
                    'ymin': float(box.xyxy[0][1]),
                    'xmax': float(box.xyxy[0][2]),
                    'ymax': float(box.xyxy[0][3]),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0]),
                    'name': result.names[int(box.cls[0])]
                }
                detections_list.append(detection)
    
    return pd.DataFrame(detections_list)

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
            print("ℹ️ No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return False
        
        # Convertir la imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            print("❌ Error al codificar la imagen")
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

        # Enviar la foto con el caption
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files, timeout=30)

        if response.status_code != 200:
            print(f"❌ Error al enviar alerta a Telegram: {response.text}")
            return False
        else:
            print("✅ Alerta enviada a Telegram (errores detectados)")
            return True
            
    except Exception as e:
        print(f"❌ Error en send_telegram_alert: {str(e)}")
        return False

@app.route('/detect', methods=['POST'])
def detect_errors():
    global model
    
    try:
        # Verificar si el modelo está cargado
        if model is None:
            return jsonify({"error": "Modelo no disponible. Revisa los logs del servidor."}), 503
        
        # Verificar si se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        # Leer imagen
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó archivo"}), 400
            
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Archivo vacío"}), 400
            
        # Convertir a PIL Image para Ultralytics
        pil_image = Image.open(BytesIO(img_bytes))
        
        print(f"🖼️ Procesando imagen de tamaño: {pil_image.size}")
        
        # Realizar detección con Ultralytics
        results = model(pil_image)
        
        # Convertir resultados a formato pandas
        detections = process_ultralytics_results(results)
        
        print(f"🔍 Detecciones encontradas: {len(detections)}")
        
        # Obtener imagen con detecciones dibujadas
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # Verificar si hay errores y enviar alerta
        has_errors = not detections[detections['name'].str.lower() != 'imprimiendo'].empty if not detections.empty else False
        alert_sent = False
        
        if has_errors:
            alert_sent = send_telegram_alert(result_img, detections)
        
        # Preparar respuesta
        response = {
            "detections": detections.to_dict(orient='records') if not detections.empty else [],
            "has_errors": has_errors,
            "alert_sent": alert_sent,
            "total_detections": len(detections)
        }
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Error procesando imagen: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    global model
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "message": "Modelo cargado correctamente" if model is not None else "Error al cargar el modelo"
    })

# Inicializar modelo al arrancar
print("🚀 Iniciando aplicación...")
model = load_model()

if model is None:
    print("⚠️ ADVERTENCIA: La aplicación se iniciará pero el modelo no está disponible")
else:
    print("✅ Aplicación iniciada correctamente")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Iniciando servidor en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)