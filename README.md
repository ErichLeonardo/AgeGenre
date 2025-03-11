# Detección de Edad y Género

Este repositorio contiene un sistema basado en OpenCV y redes neuronales para detectar rostros en imágenes o en tiempo real, estimar la edad y el género de las personas, y aplicar un desenfoque en los rostros de los menores de edad.

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install opencv-python numpy
```

## Archivos del Repositorio

- `DetectarNiñosImagenDada.py`: Script para detectar edad y género en una imagen dada.
- `DetectarNiñosTiempoReal.py`: Script para detectar edad y género en tiempo real mediante la cámara.
- `age_deploy.prototxt` y `age_net.caffemodel`: Modelo para la estimación de la edad.
- `gender_deploy.prototxt` y `gender_net.caffemodel`: Modelo para la estimación del género.
- `opencv_face_detector.pbtxt` y `opencv_face_detector_uint8.pb`: Modelos para la detección de rostros en OpenCV.

## Uso

### Detección en una imagen

Ejecutar el siguiente comando para analizar una imagen:

```bash
python DetectarNiñosImagenDada.py --image ruta/de/la/imagen.jpg
```

### Detección en tiempo real

Para ejecutar la detección de edad y género en tiempo real mediante la cámara web:

```bash
python DetectarNiñosTiempoReal.py
```

## Funcionamiento

1. **Detección de Rostros**: Se usa un modelo de OpenCV para localizar rostros en la imagen o video.
2. **Predicción de Edad y Género**: Se utiliza un modelo de redes neuronales preentrenado.
3. **Desenfoque de Rostros Menores**: Si la persona es menor de 18 años, se aplica un efecto de desenfoque en su rostro.
