# El Detector Mágico

> Proyecto Final de Inteligencia Artificial — Reconocimiento de objetos en tiempo real mediante cámara web e Inteligencia Artificial.

---

##  Descripción

Este sistema reconoce **5 tipos de objetos cotidianos** en tiempo real a través de la cámara web:
**Botella**
**Cuaderno**
**Llaves**
**Manzana**
**Taza**

Fue desarrollado utilizando **Transfer Learning** con redes neuronales convolucionales, aprovechando el modelo MobileNetV2 pre-entrenado sobre millones de imágenes.

---

## Arquitectura y Transfer Learning

### Arquitectura base elegida
Se utilizó **MobileNetV2**, una red neuronal diseñada específicamente para funcionar en dispositivos con recursos limitados. Fue pre-entrenada con millones de imágenes de la base de datos **ImageNet**, por lo que ya reconoce de forma general formas, bordes, texturas y colores.

### Justificación
Se eligió MobileNetV2 porque es **ligera, rápida y eficiente**, lo que permite que las predicciones se actualicen en tiempo real desde la cámara web sin demoras. Además, al aprovechar su conocimiento previo, no se necesita entrenar una red desde cero ni contar con miles de fotografías propias para obtener buenos resultados.

### Proceso aplicado
Se aplicó **Transfer Learning en modalidad de Extracción de Características**:
Se **congeló** toda la red MobileNetV2 para conservar su aprendizaje
Se agregaron capas nuevas al final: agrupamiento, capa densa de 128 neuronas, capa de regularización y capa de salida
Se **entrenaron solo las capas nuevas**, dejando intacta la red base

## Requisitos e Instalación

Se necesita tener instalado **Python 3.x** y las siguientes librerías:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn

## Ejecución

python entrenar.py
python camara.py

## Autor

**Gabriela Margarita Naula Castro**
Proyecto Final — Inteligencia Artificial
