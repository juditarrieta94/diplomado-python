'''API para el diplomado de Python de la Universidad de Córdoba'''


from fastapi import FastAPI
from typing import List
from classes import ModelInput, ModelOutput, APIModelBackEnd
# Creamos el objeto app
app = FastAPI(title="API EQUIPO 2 DIPLOMADO", version="1.0.0")
'''Objeto FastAPI usado para el deployment de la API :)'''
# Con el decorador, ponemos en el endpoint /predict la funcionalidad de la función predict_proba
# response_model=List[ModelOuput] es que puede responder una lista de instancias válidas de ModelOutput
# En la definición, le decimos que los Inputs son una lista de ModelInput.
# Así, la API recibe para hacer multiples predicciones


@app.post("/predict", response_model=List[ModelOutput])
async def predict(inputs: List[ModelInput]):
    """Endpoint de predicción de la API
    @param inputs: Inputs del modelo de predicción
    """
    # Creamos una lista vacía con las respuestas
    response = list()
    # Iteramos por todas las entradas que damos
    for Input in inputs:
        # Usamos nuestra Clase en el backenp para predecir con nuestros inputs.
        # Esta sería la línea que cambiamos en este archivo, podemos los inputs que necesitemos.
        # Esto es, poner Input.Nombre_Atributo
        model = APIModelBackEnd(
            Input.year, Input.month, Input.day 
        )
        response.append(model.predict()[0])
    # Retorna  la lista con todas las predicciones hechas.
    return response
