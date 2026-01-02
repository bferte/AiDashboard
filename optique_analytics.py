import pandas as pd

# Chemin vers le fichier de données
DATA_FILE = "performances_optique.csv"

def load_data():
    """Charge les données de performance depuis le fichier CSV."""
    try:
        # Spécifier le séparateur et le moteur pour éviter les avertissements
        df = pd.read_csv(DATA_FILE, sep=',', engine='python')
        # Nettoyer les espaces blancs dans les noms de colonnes
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {DATA_FILE} est introuvable.")
        return None

def get_collaborateurs_list():
    """Retourne la liste des noms des collaborateurs (excluant la moyenne)."""
    df = load_data()
    if df is not None:
        # Filtre pour exclure 'MOYENNE_MAGASIN' et retourne la liste
        collaborateurs = df[df['Collaborateur'] != 'MOYENNE_MAGASIN']['Collaborateur'].tolist()
        return collaborateurs
    return []

def get_store_average():
    """Retourne les KPIs moyens du magasin."""
    df = load_data()
    if df is not None:
        # Récupère la ligne de la moyenne et la convertit en dictionnaire
        average_data = df[df['Collaborateur'] == 'MOYENNE_MAGASIN'].iloc[0].to_dict()
        return average_data
    return {}

def get_collaborateur_data(nom: str):
    """Retourne les KPIs pour un collaborateur spécifique."""
    df = load_data()
    if df is not None:
        # Récupère la ligne du collaborateur et la convertit en dictionnaire
        collaborateur_data = df[df['Collaborateur'] == nom].iloc[0].to_dict()
        return collaborateur_data
    return {}

# Exemple d'utilisation (pour tester le module directement)
if __name__ == "__main__":
    print("Liste des collaborateurs :")
    print(get_collaborateurs_list())

    print("\nMoyenne du magasin :")
    print(get_store_average())

    print("\nDonnées pour 'Marie':")
    print(get_collaborateur_data("Marie"))
