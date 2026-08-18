import cv2
import numpy as np
import tensorflow as tf


# Cargar el modelo que entrené
modelo = tf.keras.models.load_model('detector_magico_model.h5')

# Lista de objetos que el modelo reconoce
objetos = ['botella', 'cuaderno', 'llaves', 'manzana', 'taza']

# Activar la cámara web
camara = cv2.VideoCapture(0)

# Ajustar resolución de la imagen
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Cámara lista. Pulsa 'q' para cerrar.")

while camara.isOpened():
    _, imagen = camara.read()
    if imagen is None:
        break

    # Preparar la imagen para que el modelo la entienda
    imagen_redim = cv2.resize(imagen, (224, 224))
    imagen_rgb = cv2.cvtColor(imagen_redim, cv2.COLOR_BGR2RGB)
    datos = tf.keras.applications.mobilenet_v2.preprocess_input(imagen_rgb)
    datos = np.expand_dims(datos, axis=0)

    # Hacer la predicción
    resultados = modelo.predict(datos, verbose=0)
    posicion = np.argmax(resultados[0])
    porcentaje = resultados[0][posicion] * 100
    nombre = objetos[posicion]

    # Definir color según confianza
    if porcentaje > 70:
        color_texto = (0, 255, 0)
    else:
        color_texto = (0, 165, 255)

    # Mostrar el resultado principal
    mensaje = f"{nombre}: {porcentaje:.1f}%"
    cv2.rectangle(imagen, (10, 10), (420, 60), (0, 0, 0), -1)
    cv2.putText(imagen, mensaje, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_texto, 2)

    # Mostrar las otras posibilidades
    mas_probables = np.argsort(resultados[0])[-3:][::-1]
    posicion_y = 85
    for i in mas_probables[1:]:
        otro_nombre = objetos[i]
        otro_porcentaje = resultados[0][i] * 100
        cv2.putText(imagen, f"  {otro_nombre}: {otro_porcentaje:.1f}%",
                    (20, posicion_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)
        posicion_y += 30

    # Agrandar la ventana
    imagen = cv2.resize(imagen, (1000, 700))

    # Abrir ventana con la imagen
    cv2.imshow("El Detector Magico", imagen)

    # Salir con la tecla Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()