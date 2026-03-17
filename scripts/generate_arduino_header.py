import os

# CONFIGURATION
TFLITE_MODEL_PATH = os.path.join("..", "models", "turbulence_model_quantized.tflite")
HEADER_FILE_PATH = os.path.join("..", "models", "turbulence_model_data.h")

def convert_tflite_to_c_header(tflite_path, header_path):
    if not os.path.exists(tflite_path):
        print(f"Erreur : Le fichier {tflite_path} n.est pas trouvé.")
        return

    # Lire le binaire
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()

    # Générer le contenu du header C++
    hex_lines = []
    for i, byte in enumerate(tflite_content):
        if i % 12 == 0:
            hex_lines.append("\n  ")
        hex_lines.append(f"0x{byte:02x}, ")

    header_content = f"""/* 
 * TurbulenceWatch - AI Model Data Header
 * This file was automatically generated for TinyML deployment.
 * Model size: {len(tflite_content)} bytes
 */

#ifndef TURBULENCE_MODEL_DATA_H
#define TURBULENCE_MODEL_DATA_H

const unsigned char g_turbulence_model_data[] = {{
{"".join(hex_lines)}
}};

const int g_turbulence_model_data_len = {len(tflite_content)};

#endif // TURBULENCE_MODEL_DATA_H
"""

    # Sauvegarder
    with open(header_path, 'w') as f:
        f.write(header_content)

    print(f"✅ Header C++ généré avec succès : {header_path}")
    print(f"Taille : {len(tflite_content)} octets")

if __name__ == "__main__":
    convert_tflite_to_c_header(TFLITE_MODEL_PATH, HEADER_FILE_PATH)
