import os
import subprocess
import sys

def run_step(script_name):
    print(f"\n[PIPELINE] Exécution de {script_name}...")
    try:
        # On assume que les scripts sont dans le dossier 'scripts'
        script_path = os.path.join("scripts", script_name)
        if not os.path.exists(script_path):
            print(f"[ERREUR] Le script {script_path} n'existe pas.")
            return False
            
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] Échec de {script_name}: {e}")
        return False

def main():
    print("==========================================")
    print("   TURBULENCE WATCH - RESEARCH PIPELINE   ")
    print("==========================================")
    
    # Étape 1 : Génération des données
    if run_step("generate_turbulence_dataset.py"):
        print("[SUCCÈS] Étape 1 terminée.")
    else:
        sys.exit(1)

    # Étape 2 : Exploration (Optionnel pour le pipeline auto)
    # run_step("explore_data.py")

    # Étape 3 : Extraction de features
    if run_step("extract_features.py"):
        print("[SUCCÈS] Étape 3 terminée.")
    else:
        sys.exit(1)

    # Étape 4 : Entraînement
    if run_step("train_baseline_models.py"):
        print("[SUCCÈS] Étape 4 terminée.")
    else:
        sys.exit(1)

    print("\n[FIN] Le pipeline a été exécuté avec succès.")

if __name__ == "__main__":
    main()
