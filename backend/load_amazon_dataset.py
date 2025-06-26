#!/usr/bin/env python3
"""
Dataset Amazon Polarity - Téléchargement et chargement du dataset complet
Télécharge le vrai dataset Amazon Polarity (3.6M train + 400K test)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_amazon_polarity_dataset(force_download: bool = False) -> bool:
    """
    Télécharge le dataset Amazon Polarity complet depuis Hugging Face
    
    Args:
        force_download: Force le re-téléchargement même si les fichiers existent
    
    Returns:
        bool: True si le téléchargement réussit, False sinon
    """
    try:
        from datasets import load_dataset
        
        # Chemins des fichiers
        train_path = "data/amazon_polarity/train.csv"
        test_path = "data/amazon_polarity/test.csv"
        
        # Vérifier si les fichiers existent déjà et sont volumineux
        if not force_download:
            if (os.path.exists(train_path) and os.path.exists(test_path)):
                train_size = os.path.getsize(train_path) / (1024 * 1024)  # MB
                test_size = os.path.getsize(test_path) / (1024 * 1024)   # MB
                
                # Si les fichiers sont volumineux (dataset complet), pas besoin de re-télécharger  
                if train_size > 100 and test_size > 20:  # Au moins 100MB train et 20MB test
                    logger.info(f"✅ Dataset complet déjà présent (Train: {train_size:.1f}MB, Test: {test_size:.1f}MB)")
                    return True
        
        logger.info("📥 Téléchargement du dataset Amazon Polarity complet...")
        logger.info("⚠️  ATTENTION: Le dataset complet fait ~500MB, cela peut prendre plusieurs minutes")
        
        # Créer le répertoire si nécessaire
        os.makedirs("data/amazon_polarity", exist_ok=True)
        
        # Télécharger le dataset complet
        dataset = load_dataset("amazon_polarity")
        
        # Convertir en DataFrames
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Mapper les labels (0,1) vers (1,2) pour correspondre au format attendu
        train_df['label'] = train_df['label'] + 1
        test_df['label'] = test_df['label'] + 1
        
        # Renommer les colonnes pour correspondre au format attendu
        train_df = train_df.rename(columns={'content': 'text'})
        test_df = test_df.rename(columns={'content': 'text'})
        
        # Sauvegarder les fichiers CSV
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✅ Dataset téléchargé avec succès!")
        logger.info(f"📊 Train: {len(train_df):,} échantillons")
        logger.info(f"📊 Test: {len(test_df):,} échantillons")
        logger.info(f"💾 Taille train: {os.path.getsize(train_path) / (1024*1024):.1f}MB")
        logger.info(f"💾 Taille test: {os.path.getsize(test_path) / (1024*1024):.1f}MB")
        
        return True
        
    except ImportError:
        logger.error("❌ La librairie 'datasets' n'est pas installée")
        logger.error("💡 Installez avec: pip install datasets")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement: {e}")
        return False

def load_amazon_polarity_dataset(
    max_samples: int = None,
    split: str = "train",
    random_sample: bool = False,
    use_full_dataset: bool = True
) -> List[Dict]:
    """
    Charge le dataset Amazon Polarity
    
    Args:
        max_samples: Nombre maximum d'échantillons à charger (None = tous)
        split: 'train', 'test', ou 'all'
        random_sample: Si True, échantillonnage aléatoire
        use_full_dataset: Si True, tente de télécharger le dataset complet
    
    Returns:
        List[Dict]: Liste des échantillons avec 'text' et 'label'
    """
    
    # Tentative de téléchargement du dataset complet
    if use_full_dataset:
        download_success = download_amazon_polarity_dataset()
        if not download_success:
            logger.warning("⚠️  Échec du téléchargement, utilisation du dataset de fallback")
            return load_fallback_dataset(max_samples, split, random_sample)
    
    try:
        # Chemins des fichiers
        train_path = "data/amazon_polarity/train.csv"
        test_path = "data/amazon_polarity/test.csv"
        
        # Charger les données selon le split demandé
        if split == "train":
            if not os.path.exists(train_path):
                logger.warning("❌ Fichier train.csv non trouvé, utilisation du fallback")
                return load_fallback_dataset(max_samples, split, random_sample)
            df = pd.read_csv(train_path)
        elif split == "test":
            if not os.path.exists(test_path):
                logger.warning("❌ Fichier test.csv non trouvé, utilisation du fallback")
                return load_fallback_dataset(max_samples, split, random_sample)
            df = pd.read_csv(test_path)
        elif split == "all":
            dfs = []
            if os.path.exists(train_path):
                dfs.append(pd.read_csv(train_path))
            if os.path.exists(test_path):
                dfs.append(pd.read_csv(test_path))
            if not dfs:
                logger.warning("❌ Aucun fichier trouvé, utilisation du fallback")
                return load_fallback_dataset(max_samples, split, random_sample)
            df = pd.concat(dfs, ignore_index=True)
        else:
            logger.error(f"❌ Split invalide: {split}")
            return []
        
        # Vérifier les colonnes nécessaires
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error("❌ Colonnes 'text' et 'label' requises")
            return load_fallback_dataset(max_samples, split, random_sample)
        
        # Nettoyer les données
        df = df.dropna(subset=['text', 'label'])
        
        # Échantillonnage si demandé
        if max_samples and len(df) > max_samples:
            if random_sample:
                df = df.sample(n=max_samples, random_state=42)
            else:
                df = df.head(max_samples)
        
        # Convertir en format attendu
        samples = []
        for _, row in df.iterrows():
            samples.append({
                'text': str(row['text']),
                'label': int(row['label'])
            })
        
        logger.info(f"✅ Dataset chargé: {len(samples):,} échantillons ({split})")
        
        # Statistiques
        if samples:
            labels = [s['label'] for s in samples]
            pos_count = sum(1 for l in labels if l == 2)
            neg_count = sum(1 for l in labels if l == 1)
            logger.info(f"📊 Positifs: {pos_count:,} | Négatifs: {neg_count:,}")
        
        return samples
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return load_fallback_dataset(max_samples, split, random_sample)

def load_fallback_dataset(
    max_samples: int = None,
    split: str = "train", 
    random_sample: bool = False
) -> List[Dict]:
    """
    Dataset de fallback si le téléchargement échoue
    """
    logger.info("🔄 Utilisation du dataset de fallback...")
    
    # Données de base pour les tests
    fallback_samples = [
        {"text": "This product is amazing! Great quality and fast shipping.", "label": 2},
        {"text": "Excellent purchase, highly recommend to everyone.", "label": 2},
        {"text": "Perfect item, exactly as described.", "label": 2},
        {"text": "Outstanding quality, exceeded my expectations.", "label": 2},
        {"text": "Fantastic product, will buy again.", "label": 2},
        {"text": "Terrible quality, broke after one day.", "label": 1},
        {"text": "Worst purchase ever, complete waste of money.", "label": 1},
        {"text": "Poor quality materials, very disappointed.", "label": 1},
        {"text": "Awful product, doesn't work as advertised.", "label": 1},
        {"text": "Horrible experience, would not recommend.", "label": 1},
    ]
    
    # Dupliquer pour avoir plus de données si nécessaire
    if max_samples and max_samples > len(fallback_samples):
        multiplier = (max_samples // len(fallback_samples)) + 1
        fallback_samples = fallback_samples * multiplier
    
    # Appliquer les filtres
    if max_samples:
        if random_sample:
            np.random.seed(42)
            indices = np.random.choice(len(fallback_samples), min(max_samples, len(fallback_samples)), replace=False)
            fallback_samples = [fallback_samples[i] for i in indices]
        else:
            fallback_samples = fallback_samples[:max_samples]
    
    logger.info(f"✅ Dataset fallback: {len(fallback_samples)} échantillons")
    return fallback_samples

def get_dataset_info() -> Dict:
    """
    Retourne des informations sur le dataset disponible
    """
    train_path = "data/amazon_polarity/train.csv"
    test_path = "data/amazon_polarity/test.csv"
    
    info = {
        "train_exists": os.path.exists(train_path),
        "test_exists": os.path.exists(test_path),
        "train_size": 0,
        "test_size": 0,
        "train_file_size_mb": 0,
        "test_file_size_mb": 0,
        "is_full_dataset": False
    }
    
    if info["train_exists"]:
        try:
            df = pd.read_csv(train_path)
            info["train_size"] = len(df)
            info["train_file_size_mb"] = os.path.getsize(train_path) / (1024 * 1024)
        except:
            pass
    
    if info["test_exists"]:
        try:
            df = pd.read_csv(test_path)
            info["test_size"] = len(df)
            info["test_file_size_mb"] = os.path.getsize(test_path) / (1024 * 1024)
        except:
            pass
    
    # Déterminer si c'est le dataset complet (heuristique basée sur la taille)
    info["is_full_dataset"] = (
        info["train_size"] > 1000000 and  # Plus d'1M d'échantillons d'entraînement
        info["test_size"] > 100000        # Plus de 100K d'échantillons de test
    )
    
    return info

class AmazonDatasetLoader:
    """
    Classe pour charger le dataset Amazon Polarity complet
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, split: str = "train", max_samples: int = None, random_sample: bool = False) -> List[str]:
        """
        Charge les textes du dataset (format simple pour compatibilité)
        
        Returns:
            List[str]: Liste des textes seulement
        """
        samples = load_amazon_polarity_dataset(max_samples, split, random_sample, use_full_dataset=True)
        return [sample['text'] for sample in samples]
    
    def load_labeled_data(self, split: str = "train", max_samples: int = None, random_sample: bool = False) -> Tuple[List[str], List[int]]:
        """
        Charge les textes et labels du dataset
        
        Returns:
            Tuple[List[str], List[int]]: (textes, labels)
        """
        samples = load_amazon_polarity_dataset(max_samples, split, random_sample, use_full_dataset=True)
        texts = [sample['text'] for sample in samples]
        labels = [sample['label'] for sample in samples]
        return texts, labels
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques du dataset
        """
        return get_dataset_info()

# Instance globale pour compatibilité
amazon_loader = AmazonDatasetLoader()

if __name__ == "__main__":
    # Test du téléchargement
    print("🚀 Test du téléchargement du dataset Amazon Polarity complet...")
    
    # Afficher les infos actuelles
    info = get_dataset_info()
    print(f"📊 État actuel:")
    print(f"   Train: {info['train_size']:,} échantillons ({info['train_file_size_mb']:.1f}MB)")
    print(f"   Test: {info['test_size']:,} échantillons ({info['test_file_size_mb']:.1f}MB)")
    print(f"   Dataset complet: {'✅' if info['is_full_dataset'] else '❌'}")
    
    # Télécharger le dataset complet
    if not info['is_full_dataset']:
        print("\n📥 Téléchargement du dataset complet...")
        success = download_amazon_polarity_dataset(force_download=True)
        
        if success:
            # Afficher les nouvelles infos
            info = get_dataset_info()
            print(f"\n✅ Téléchargement terminé!")
            print(f"📊 Nouveau état:")
            print(f"   Train: {info['train_size']:,} échantillons ({info['train_file_size_mb']:.1f}MB)")
            print(f"   Test: {info['test_size']:,} échantillons ({info['test_file_size_mb']:.1f}MB)")
            print(f"   Dataset complet: {'✅' if info['is_full_dataset'] else '❌'}")
        else:
            print("❌ Échec du téléchargement")
    else:
        print("✅ Dataset complet déjà disponible!")
    
    # Test de chargement
    print("\n🧪 Test de chargement...")
    samples = load_amazon_polarity_dataset(max_samples=10, split="train")
    print(f"📊 Échantillons chargés: {len(samples)}")
    
    if samples:
        print("\n📝 Premiers échantillons:")
        for i, sample in enumerate(samples[:3]):
            label_text = "Positif" if sample['label'] == 2 else "Négatif"
            text_preview = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
            print(f"   {i+1}. [{label_text}] {text_preview}") 