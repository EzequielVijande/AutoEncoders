# Neural Network Base

Implementaciones de Auto-encoders y Variational Auto-encoders (VAE) en Python basada en implementación propia de redes neuronales, diseñada para ser fácilmente extensible para diversos proyectos de reconocimiento y clasificación.

## Características

- Arquitectura modular con clases para Layer y NeuralNetwork.
- Soporte para múltiples funciones de activación: ReLU, Sigmoid, Softmax.
- Implementación de la función de pérdida de entropía cruzada, MSE y MAE.
- Entrenamiento por lotes con Descenso de Gradiente Estocástico (SGD), SGD + momento y Adam.
- Funcionalidad para guardar y cargar pesos de la red.
- Ejemplos de uso.
- Pruebas unitarias para asegurar la integridad del código.

## Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/EzequielVijande/AutoEncoders.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd AutoEncoders
    ```
3. (Opcional) Crea un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
4. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

