"""
Revisa todos los archivos del proyecto y avisa si aparece contenido obsoleto.
Uso: python revisar_contenido.py
"""
import os
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_TERMINOS = os.path.join(CARPETA, "terminos_prohibidos.txt")
EXTENSIONES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".html", ".js", ".csv"}
IGNORAR_CARPETAS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
IGNORAR_ARCHIVOS = {"terminos_prohibidos.txt", "revisar_contenido.py"}


def cargar_terminos():
    terminos = []
    with open(ARCHIVO_TERMINOS, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                terminos.append(linea.lower())
    return terminos


def revisar():
    terminos = cargar_terminos()
    hallazgos = 0
    for raiz, carpetas, archivos in os.walk(CARPETA):
        carpetas[:] = [c for c in carpetas if c not in IGNORAR_CARPETAS]
        for nombre in archivos:
            ext = os.path.splitext(nombre)[1].lower()
            if ext not in EXTENSIONES or nombre in IGNORAR_ARCHIVOS:
                continue
            ruta = os.path.join(raiz, nombre)
            try:
                with open(ruta, encoding="utf-8") as f:
                    lineas = f.readlines()
            except (UnicodeDecodeError, OSError):
                continue
            for n, linea in enumerate(lineas, start=1):
                texto = linea.lower()
                for termino in terminos:
                    if termino in texto:
                        hallazgos += 1
                        print(f"{os.path.relpath(ruta, CARPETA)}, línea {n}: '{termino}'")
    if hallazgos == 0:
        print("OK: no se encontró contenido obsoleto.")
        return 0
    print(f"\nSe encontraron {hallazgos} coincidencias. Revísalas: bórralas o reescríbelas.")
    print("Nota: algunas pueden ser falsos positivos (por ejemplo, 'evaluación de impacto' en un texto neutral).")
    return 1


if __name__ == "__main__":
    sys.exit(revisar())
