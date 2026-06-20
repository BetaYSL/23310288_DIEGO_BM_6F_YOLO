# Proyecto de Visión Artificial – Detección de Celulares con YOLOv8

## Integrante

DIEGO BETANCOURT MÉNDEZ 6F 23310288

---

# Descripción del Proyecto

Este proyecto consiste en el desarrollo de un sistema de visión artificial basado en el modelo **YOLOv8 (You Only Look Once)** para la detección automática de teléfonos celulares en imágenes.

El objetivo principal es demostrar el funcionamiento del entrenamiento, ajuste y aplicación de un modelo de detección de objetos utilizando aprendizaje profundo y procesamiento de imágenes.

El desarrollo fue realizado utilizando:

* Python
* Google Colab
* Roboflow
* YOLOv8 (Ultralytics)
* PyTorch

---

# Objetivos

## Objetivo General

Desarrollar un modelo de visión artificial capaz de detectar teléfonos celulares mediante técnicas de aprendizaje profundo.

---

## Objetivos Específicos

* Construir un conjunto de datos personalizado.
* Etiquetar imágenes utilizando Roboflow.
* Entrenar un modelo YOLOv8.
* Evaluar el desempeño del modelo.
* Realizar inferencia sobre nuevas imágenes.

---

# Herramientas Utilizadas

| Herramienta  | Función                   |
| ------------ | ------------------------- |
| Python       | Desarrollo del sistema    |
| Google Colab | Entrenamiento             |
| Roboflow     | Preparación del dataset   |
| YOLOv8       | Detección de objetos      |
| OpenCV       | Procesamiento de imágenes |
| PyTorch      | Motor de entrenamiento    |

---

# Dataset Utilizado

Dataset personalizado generado mediante Roboflow.

## Clase Entrenada

```text
celular
```

## Configuración utilizada

```text
Epochs: 30
Resolución: 640 × 640
Modelo: YOLOv8 Nano
```

---

# Resultados del Entrenamiento

Resultados obtenidos durante el entrenamiento:

```text
Precisión (P): 98.5%
Recall (R): 78.3%
mAP50: 91.2%
mAP50–95: 81.7%
```

Interpretación:

El modelo logró detectar teléfonos celulares con un nivel alto de precisión considerando el tamaño del conjunto de datos utilizado.


---

# Instalación

Instalar dependencias:

```bash
pip install ultralytics
```

---

# Ejecución

Ejecutar el notebook en Google Colab.

Posteriormente:

```bash
Entrenar → Generar best.pt → Ejecutar inferencia
```

---

# Código Fuente

## 1. Instalación de Dependencias

```python
!pip install ultralytics
```

---

## 2. Importación del Modelo

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("Modelo cargado correctamente")
```

---

## 3. Carga del Dataset

```python
from google.colab import files

uploaded = files.upload()
```

---

## 4. Descompresión

```python
!unzip "CelularesYOLO.v3i.yolov8.zip"
```

---

## 5. Verificación

```python
!ls
```

---

## 6. Verificación GPU

```python
import torch

print(torch.cuda.is_available())
```

---

## 7. Entrenamiento

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/content/data.yaml",
    epochs=30,
    imgsz=640
)
```

---

## 8. Cargar Modelo Entrenado

```python
modelo = YOLO(
"/content/runs/detect/train-3/weights/best.pt"
)
```

---

## 9. Cargar Imagen

```python
from google.colab import files

files.upload()
```

---

## 10. Detección

```python
from PIL import Image

resultado = modelo(
"/content/imagen_prueba.jpg"
)

for r in resultado:

    imagen = Image.fromarray(
        r.plot()[...,::-1]
    )

    display(imagen)
```

---

# Caso de Estudio

Implementación de un Sistema de Visión Artificial para la Detección Automática de Teléfonos Celulares en Entornos Industriales utilizando YOLOv8

La visión artificial se ha convertido en una herramienta ampliamente utilizada en procesos industriales debido a su capacidad para automatizar tareas de inspección, monitoreo y control mediante el análisis de imágenes en tiempo real.

Dentro de muchos entornos industriales y operativos, uno de los problemas más frecuentes es el uso indebido de dispositivos móviles por parte del personal durante actividades críticas. Aunque los teléfonos celulares son herramientas de comunicación importantes, su uso durante la operación de maquinaria o ejecución de tareas delicadas puede generar riesgos operativos y afectar el desempeño general del proceso.

Para atender esta problemática se desarrolló un sistema basado en inteligencia artificial utilizando el modelo YOLO (You Only Look Once), entrenado específicamente para detectar teléfonos celulares en imágenes.

El objetivo del sistema es identificar automáticamente la presencia de celulares y generar una respuesta inmediata que ayude a mantener condiciones de trabajo más seguras y controladas.

Problema a Resolver

En numerosas áreas industriales, el uso del teléfono celular durante jornadas operativas representa una fuente potencial de:

Distracción del operador.
Reducción de productividad.
Incremento del tiempo muerto.
Mayor probabilidad de accidentes.
Incumplimiento de normas internas de seguridad.

Actualmente muchas empresas dependen de supervisión humana para detectar estas situaciones.

Este método presenta desventajas importantes:

Supervisión limitada.
Costos elevados.
Fatiga del personal.
Baja capacidad de monitoreo continuo.

Por ello se propone automatizar este proceso mediante visión artificial.

Objetivo del Sistema

Diseñar e implementar un sistema de detección automática de teléfonos celulares utilizando un modelo YOLO entrenado con imágenes etiquetadas manualmente.

El sistema deberá:

Capturar imágenes continuamente.
Analizar cada imagen utilizando el modelo entrenado.
Detectar la presencia de teléfonos celulares.
Generar una acción automática ante una detección positiva.

Descripción General de la Solución

La solución propuesta consiste en instalar cámaras estratégicamente ubicadas dentro del área operativa.

Cada imagen capturada será enviada a una estación de procesamiento donde se ejecutará el modelo de detección.

Cuando el modelo identifique un celular:

Dibujará una caja delimitadora (bounding box).
Registrará hora y evento.
Generará una alerta al supervisor.
Almacenará evidencia para auditorías.

Arquitectura del Sistema
Etapa 1 — Captura de Imagen

Se instala una cámara de monitoreo sobre el área de trabajo.

Características propuestas:

Resolución mínima: 1080p.
Frecuencia: 30 FPS.
Ángulo amplio.

Entrada:
Escena industrial
↓
Captura de video

Etapa 2 — Procesamiento

El video se envía al equipo de procesamiento.

Hardware propuesto:
| Componente     | Especificación     |
| -------------- | ------------------ |
| Procesador     | Intel i7 / Ryzen 7 |
| GPU            | NVIDIA RTX o Tesla |
| RAM            | 16 GB              |
| Almacenamiento | SSD 512 GB         |

Software:

Python
YOLOv8
OpenCV
Google Colab (entrenamiento)
Roboflow

Proceso:
Imagen
↓

Preprocesamiento

↓

Modelo YOLO

↓

Detección

Etapa 3 — Generación de Acción

Si el sistema detecta un celular:

Condición:
Confianza ≥ 80%

Acciones posibles:

Encender indicador visual.
Generar alarma sonora.
Registrar evidencia.
Enviar notificación.

Funcionamiento del Modelo Entrenado

Para este proyecto se construyó un conjunto de datos personalizado.

Proceso realizado:

Recolección de imágenes.
Etiquetado manual en Roboflow.
Creación del dataset.
Exportación YOLOv8.
Entrenamiento en Google Colab.

Clase utilizada:
celular

Modelo empleado:
YOLOv8 Nano

Resultado obtenido:
mAP50 ≈ 91%

Interpretación:

El modelo logra detectar correctamente teléfonos celulares con un nivel alto de precisión para un conjunto de prueba pequeño.

Flujo Completo de Operación
Cámara

↓

Captura imagen

↓

Procesamiento YOLO

↓

¿Detectó celular?

↓ SI

Generar alerta

↓

Guardar evidencia

↓

Enviar notificación

↓ NO

Continuar monitoreo

Aplicaciones Potenciales

Este sistema podría implementarse en:

Líneas de manufactura.
Plantas automotrices.
Centros logísticos.
Laboratorios.
Almacenes.
Centros de distribución.
Áreas con maquinaria automatizada.

Limitaciones del Proyecto

Aunque el sistema obtuvo resultados satisfactorios, existen factores que pueden afectar el desempeño:

Baja iluminación.
Objetos similares a celulares.
Teléfonos parcialmente ocultos.
Reflejos.
Variación extrema de posiciones.

Para mejorar el sistema se recomienda:

Incrementar el tamaño del dataset.
Entrenar más épocas.
Agregar imágenes reales industriales.
Incorporar múltiples clases.

Este proyecto permitió aplicar conceptos de visión artificial mediante el entrenamiento de un modelo YOLO para detección de objetos.

Los resultados demuestran que es posible construir sistemas automatizados capaces de identificar teléfonos celulares en imágenes con una precisión adecuada para escenarios reales.

La integración de esta tecnología podría contribuir a mejorar la seguridad operativa, reducir supervisión manual y facilitar la automatización de procesos industriales.



---

# Autor

DIEGO BETANCOURT MÉNDEZ

Proyecto desarrollado para la materia de Visión Artificial.
