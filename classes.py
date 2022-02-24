from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from typing import Literal
import joblib
import pandas as pd
from fastapi import HTTPException
import numpy as np

# Entrada del modelo.

# Adapten esta clase según cómo sean sus entradas.
# Miren su X, ¿cómo eran los datos con los que aprendió el modelo?
# Recuerden que las variables categoricas acá las dejaremos en un solo parametro.
# Miren cómo trabajamos con el day para un ejemplo de variable categorica.

# Una variable binaria puede ser:
# binaria: int = Field(ge=0, le=1)
# O, binaria: bool

class ModelInput(PydanticBaseModel):
    """
    Clase que define las entradas del modelo
    """

    year: int = Field(
        description="Horas promedio trabajadas al mes", ge=2005, le=2025
    )
    month: int = Field(
        description="Horas promedio trabajadas al mes", ge=1, le=12
    )
    day: int = Field(
        description="Horas promedio trabajadas al mes", ge=1, le=31
    )

    # OPCIONAL: Poner el ejemplo para que en la documentación ya puedan de una lanzar la predicción.
    class Config:
        schema_extra = {
            "example": {
                "year": 2022,
                "month": 2,
                "day": 5,
            }
        }


class ModelOutput(PydanticBaseModel):
    """
    Clase que define las salidas del modelo
    """

    cantidad_empresas: float = Field(
        description="Cantidad de empresas que se registraran para esa fecha"
    )

    class Config:
        schema_extra = {"example": {"cantidad_empresas": 22}}


class APIModelBackEnd:
    """
    Esta clase maneja el back end de nuestro modelo de Machine Learning para la API en FastAPI
    """

    def __init__(self, year: int, month: int, day: int):
        """
        Este método se usa al instanciar las clases
        Aquí, hacemos que pida los mismos parámetros que tenemos en ModelInput.
        Para más información del __init__ method, pueden leer en línea en sitios cómo
        https://www.udacity.com/blog/2021/11/__init__-in-python-an-overview.html
        Este método lo cambian según sus inputs
        @param year: año
        @param month: mes
        @param day: dia
        """
        self.year = year
        """año"""
        self.month = month
        """mes"""
        self.day = day
        """dia"""

        self.model= None
        """Modelo de ML cargado por el método L{_load_model}"""

    def _load_model(self, model_filename: str = "modelo.pkl"):
        """
        Clase para cargar el modelo. Es una forma exótica de correr joblib.load pero teniendo funcionalidad con la API.
        Este método seguramente no lo van a cambiar, y si lo cambian, cambian el valor por defecto del string
        @param model_filename: Nombre del modelo en formato .pkl
        """
        # Asignamos a un atributo el nombre del archivo
        self.model_filename = model_filename
        """Nombre del modelo"""
        try:
            # Se intenta cargar el modelo
            self.model = joblib.load(self.model_filename)
        except Exception:
            # Si hay un error, se levanda una Exception de HTTP diciendo que no se encontró el modelo
            raise HTTPException(
                status_code=404,
                detail=f"Modelo con el nombre {self.model_filename} no fue encontrado",
            )
        # Si todo corre ok, imprimimos que cargamos el modelo
        print(f"El modelo '{self.model_filename}' fue cargado exitosamente")

    def _prepare_data(self):
        """
        Clase de preparar lo datos.
        Este método convierte las entradas en los datos que tenían en X_train y X_test.
        Miren el orden de las columnas de los datos antes de su modelo.
        Tienen que recrear ese orden, en un dataframe de una fila.
        """
        # Aquí, ponemos los valores para los niveles de satisfacción.
        # Pueden manejar así las variables categoricas.
        # Revisen los X!!! De eso depende que valores hay aquí.
        # Para ver más o menos que valores pueden ser, en un data frame se le aplico pd.get_dummies, corran algo como:
        # X_test[[col for col in X_test.columns if "nombre de columna" in col]].drop_duplicates()

        

        # Hacemos el DataFrame.
        # Ponemos en columns lo que nos da de correr list(X_test.columns)
        # En data, ponemos los datos en el orden en que están en las columnas

        df = pd.DataFrame(
            columns=[
                "year",
                "month",
                "day",
               
            ],
            data=[
                [
                    self.year,
                    self.month,
                    self.day,
                ]
            ],
        )

        # Ese * en *days[self.day] hace unpacking a la lista.
        # Sería como escribir days[self.day][0], days[self.day][1]
        return df

    def predict(self, y_name: str = "cantidad_empresas"):
        """
        Clase para predecir.
        Carga el modelo, prepara los datos y predice.
        Acá, solo deberían cambiar en el input el valor por defecto de y_name (eso en rojo que dice cantidad_empresas)
        para que sea coherente con su ModelOutput
        además de quizá, la línea
        prediction = pd.DataFrame(self.model.predict_proba(X)[:,1]).rename(columns={0:y_name})
        por
        prediction = pd.DataFrame(self.model.predict(X)).rename(columns={0:y_name})
        @param y_name: Nombre de la variable de salida del modelo
        @returns: JSON de la predicción
        """
        self._load_model()
        x = self._prepare_data()
        prediction = pd.DataFrame(np.round(self.model.predict(x))).rename(
            columns={0: y_name}
        )
        return prediction.to_dict(orient="records")
