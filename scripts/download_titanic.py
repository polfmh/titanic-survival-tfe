"""
Script de descarga del dataset Titanic desde Kaggle.

USO:
  1. Configura tu API token de Kaggle de una de estas formas:
     a) Pon el archivo kaggle.json en ~/.kaggle/kaggle.json
        (descárgalo desde https://www.kaggle.com/settings -> API -> Create New Token)
     b) O establece variables de entorno antes de ejecutar:
        set KAGGLE_USERNAME=tu_usuario
        set KAGGLE_KEY=tu_api_key

  2. Acepta las reglas de la competición en:
     https://www.kaggle.com/competitions/titanic/rules

  3. Ejecuta:
     python download_titanic.py
"""

import os
import sys
import json
import zipfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "titanic_kaggle"
OUTPUT_DIR.mkdir(exist_ok=True)

def setup_kaggle_token():
    """Configura el token de Kaggle desde variable de entorno si está disponible."""
    api_token_json = os.environ.get("KAGGLE_API_TOKEN")
    if api_token_json:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(api_token_json)
        kaggle_json.chmod(0o600)
        print("Token de Kaggle configurado desde variable de entorno.")
        return True

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        kaggle_json.chmod(0o600)
        print(f"Token de Kaggle configurado: usuario={username}")
        return True

    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        print("Token de Kaggle encontrado en ~/.kaggle/kaggle.json")
        return True

    return False

def download():
    if not setup_kaggle_token():
        print("\nERROR: No se encontró token de Kaggle.")
        print("Opciones:")
        print("  1. Descarga kaggle.json desde https://www.kaggle.com/settings")
        print("     y ponlo en: ~/.kaggle/kaggle.json")
        print("  2. Establece variables de entorno:")
        print("     KAGGLE_USERNAME=tu_usuario")
        print("     KAGGLE_KEY=tu_api_key")
        print("  3. Descarga manualmente desde:")
        print("     https://www.kaggle.com/competitions/titanic/data")
        print("     y coloca train.csv, test.csv, gender_submission.csv en:")
        print(f"     {OUTPUT_DIR}")
        sys.exit(1)

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("Instalando kaggle...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

    api = KaggleApiExtended()
    api.authenticate()

    print(f"Descargando dataset de la competición titanic en {OUTPUT_DIR}...")
    api.competition_download_files("titanic", path=str(OUTPUT_DIR), quiet=False)

    # Descomprimir si hay zip
    for zip_file in OUTPUT_DIR.glob("*.zip"):
        print(f"Descomprimiendo {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(OUTPUT_DIR)
        zip_file.unlink()

    files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\nDescarga completada. Archivos en {OUTPUT_DIR}:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    download()
