# Tesis

## Importante!

Si se desea ejecutar el codigo tener en cuenta que la implementación de Chronos se realizó en un entorno virtual distinto al del resto de modelos por problemas de dependencias de `autogluon`. Las dependencias de ambos entornos se encuentran en [`requirements.txt`](link) para la aplicación de los modelos en general y en [`requirements_autogluon.txt`](link) para Chronos.

```bash
# -------- CREACION DE LOS ENTORNOS ---------
# -------- Entorno general ---------
cd Carpeta_Del_Repositorio/

# Creacion
python -m venv .venv

# Carga del entorno
source .venv/Scripts/Activate

# Instalacion de librerias
pip install -r requirements.txt

# -------- Entorno Chronos ---------
# Creacion
python -m venv .venv_autogluon

# Carga del entorno
source .venv_autogluon/Scripts/Activate

# Instalacion de librerias
pip install -r requirements_chronos.txt

```

