import requests
from bs4 import BeautifulSoup
import os

# Liste des URLs de ton site à indexer
URLS = [
    "https://eduscol.education.fr/",
    "https://eduscol.education.fr/3076/innovation-pedagogique-et-experimentation-derogatoire",
    "https://eduscol.education.fr/3921/l-education-au-developpement-durable-dans-le-cadre-des-enseignements",
    "https://eduscol.education.fr/83/j-enseigne-au-cycle-1"
]

OUTPUT_DIR = "data/html"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def scrape_site():
    for url in URLS:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # On nettoie : on enlève scripts et styles
            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()

            # On récupère le texte propre
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            # Nom de fichier basé sur l'URL
            filename = url.split("/")[-1] or "index"
            filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Source URL: {url}\n\n{clean_text}")
                
            print(f"✅ Scrappé : {url} -> {filepath}")
            
        except Exception as e:
            print(f"❌ Erreur sur {url}: {e}")

if __name__ == "__main__":
    scrape_site()