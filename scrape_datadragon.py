"""
Scraper Data Dragon (API officielle Riot) pour les abilities détaillées.
Ajoute au fichier TXT existant.
"""
import requests
from pathlib import Path
import json
import time
import re

OUTPUT_DIR = Path("data/unstructured")
CHAMPIONS_DIR = Path("data/champions")

# Data Dragon API
DDRAGON_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/fr_FR/champion/{champion}.json"


def get_latest_version() -> str:
    """Récupère la dernière version de Data Dragon."""
    response = requests.get(DDRAGON_VERSION_URL, timeout=10)
    versions = response.json()
    return versions[0]


def clean_html(text: str) -> str:
    """Supprime les balises HTML."""
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def scrape_datadragon(champion_name: str) -> str:
    """Récupère les données d'un champion depuis Data Dragon."""
    
    # Format du nom pour l'API (Lee Sin -> LeeSin, Kai'Sa -> Kaisa)
    api_name = champion_name.replace(" ", "").replace("'", "").replace(".", "")
    
    # Cas spéciaux
    special_names = {
        "Wukong": "MonkeyKing",
        "Renata Glasc": "Renata",
        "Nunu & Willump": "Nunu",
        "Kai'Sa": "Kaisa",
        "KaiSa": "Kaisa",
    }
    api_name = special_names.get(champion_name, api_name)
    
    print(f"  → Data Dragon: {api_name}")
    
    try:
        version = get_latest_version()
        url = DDRAGON_CHAMPION_URL.format(version=version, champion=api_name)
        
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"    ✗ Status {response.status_code}")
            return ""
        
        data = response.json()
        champ_data = data["data"][api_name]
        
        texts = []
        
        # Lore
        lore = champ_data.get("lore", "")
        if lore:
            texts.append(f"## Lore\n{lore}")
        
        # Blurb (résumé)
        blurb = champ_data.get("blurb", "")
        if blurb and blurb != lore:
            texts.append(f"## Résumé\n{blurb}")
        
        # Abilities (spells)
        spells = champ_data.get("spells", [])
        passive = champ_data.get("passive", {})
        
        if passive or spells:
            texts.append("\n## Compétences détaillées")
            
            # Passive
            if passive:
                p_name = passive.get("name", "Passif")
                p_desc = clean_html(passive.get("description", ""))
                texts.append(f"\n### Passif - {p_name}\n{p_desc}")
            
            # Q, W, E, R
            keys = ["Q", "W", "E", "R"]
            for i, spell in enumerate(spells):
                key = keys[i] if i < len(keys) else f"Spell{i}"
                s_name = spell.get("name", "")
                s_desc = clean_html(spell.get("description", ""))
                s_tooltip = clean_html(spell.get("tooltip", ""))
                
                texts.append(f"\n### {key} - {s_name}")
                texts.append(s_desc)
                if s_tooltip and s_tooltip != s_desc:
                    texts.append(f"Détails: {s_tooltip[:500]}")
        
        # Tips
        ally_tips = champ_data.get("allytips", [])
        enemy_tips = champ_data.get("enemytips", [])
        
        if ally_tips:
            texts.append("\n## Conseils (alliés)")
            for tip in ally_tips:
                texts.append(f"- {tip}")
        
        if enemy_tips:
            texts.append("\n## Conseils (contre)")
            for tip in enemy_tips:
                texts.append(f"- {tip}")
        
        result = "\n\n".join(texts)
        print(f"    ✓ {len(result)} chars")
        return result
        
    except Exception as e:
        print(f"    ✗ Erreur: {e}")
        return ""


def append_to_wiki(champion_id: str, new_content: str, source_name: str) -> bool:
    """Ajoute du contenu à un fichier wiki existant."""
    wiki_path = OUTPUT_DIR / f"{champion_id}.txt"
    
    if not wiki_path.exists():
        print(f"  ✗ {wiki_path} n'existe pas")
        return False
    
    with open(wiki_path, "r", encoding="utf-8") as f:
        existing = f.read()
    
    marker = f"## {source_name}"
    if marker in existing:
        print(f"  ⏭ {source_name} déjà présent")
        return True
    
    with open(wiki_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"## {source_name}\n\n")
        f.write(new_content)
    
    print(f"  ✓ Ajouté à {champion_id}.txt")
    return True


def enrich_champion(champion_name: str, champion_id: str):
    """Enrichit un champion avec Data Dragon."""
    print(f"\n[{champion_name}]")
    
    content = scrape_datadragon(champion_name)
    
    if content and len(content) > 200:
        append_to_wiki(champion_id, content, "Data Dragon (Riot officiel)")
    else:
        print(f"  ✗ Pas assez de contenu")


def enrich_all():
    """Enrichit tous les champions du dossier unstructured."""
    txt_files = list(OUTPUT_DIR.glob("*.txt"))
    print(f"Champions à enrichir: {len(txt_files)}\n")
    
    for path in sorted(txt_files):
        champ_id = path.stem
        # Convertit l'id en nom (kaisa -> Kai'Sa, leesin -> Lee Sin)
        name = champ_id.capitalize()
        
        # Cas spéciaux
        special_names = {
            "kaisa": "Kai'Sa",
            "leesin": "Lee Sin",
            "missfortune": "Miss Fortune",
            "jarvaniv": "Jarvan IV",
            "drmundo": "Dr. Mundo",
            "masteryi": "Master Yi",
            "twistedfate": "Twisted Fate",
            "xinzhao": "Xin Zhao",
            "aurelionsol": "Aurelion Sol",
            "tahmkench": "Tahm Kench",
            "reksai": "Rek'Sai",
            "khazix": "Kha'Zix",
            "velkoz": "Vel'Koz",
            "chogath": "Cho'Gath",
            "kogmaw": "Kog'Maw",
            "belveth": "Bel'Veth",
            "renataglassc": "Renata Glasc",
            "nunuwillump": "Nunu",
            "wukong": "Wukong",
        }
        name = special_names.get(champ_id, name)
        
        enrich_champion(name, champ_id)
        time.sleep(0.3)


def enrich_single(champion_name: str):
    """Enrichit un seul champion."""
    champ_id = champion_name.lower().replace("'", "").replace(" ", "")
    enrich_champion(champion_name, champ_id)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        enrich_single(sys.argv[1])
    else:
        enrich_all()
