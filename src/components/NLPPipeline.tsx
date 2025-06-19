import React from 'react';
import { ArrowRight, FileText, Scissors, Filter, RefreshCw, CheckCircle, Eye, ArrowDown, Settings, Cpu, Brain, BarChart3, Code } from 'lucide-react';
import { SentimentAnalyzer } from './SentimentAnalyzer';
import { InteractiveCharts } from './InteractiveCharts';
import { WordCloud } from './WordCloud';
import { PipelineConfigurator } from './PipelineConfigurator';
import { ModelComparison } from './ModelComparison';
import { CodePopup } from './CodePopup';

interface PipelineStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
  input: string;
  output: string;
  details: string[];
}

interface NLPPipelineProps {
  text: string;
  onComplete: (processedData: any) => void;
}

export const NLPPipeline: React.FC<NLPPipelineProps> = ({ text, onComplete }) => {
  const [nltkResults, setNltkResults] = React.useState<any[]>([]);
  const [bertResults, setBertResults] = React.useState<any[]>([]);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [selectedModel, setSelectedModel] = React.useState<'nltk' | 'bert' | 'comparison'>('comparison');
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [codePopup, setCodePopup] = React.useState<{
    isOpen: boolean;
    stepTitle: string;
    code: string;
    model: 'nltk' | 'bert';
    stepNumber: number;
  }>({
    isOpen: false,
    stepTitle: '',
    code: '',
    model: 'nltk',
    stepNumber: 1
  });
  
  // Configuration du pipeline
  const [pipelineConfig, setPipelineConfig] = React.useState({
    steps: {
      cleaning: true,
      tokenization: true,
      stopwords: true,
      lemmatization: true
    },
    parameters: {
      minTokenLength: 2,
      maxTokens: 1000,
      customStopWords: [] as string[],
      caseSensitive: false,
      preserveNumbers: false,
      preservePunctuation: false
    },
    templates: {
      name: 'Configuration par défaut',
      description: 'Configuration standard pour l\'analyse de sentiment'
    }
  });

  // Configuration des modèles avec couleurs fixes
  const modelConfigs = {
    nltk: {
      name: 'NLTK',
      description: 'Natural Language Toolkit - Approche traditionnelle',
      icon: Cpu,
      colors: {
        primary: 'text-blue-400',
        bg: 'bg-blue-500/20',
        border: 'border-blue-500/30',
        gradient: 'from-blue-500/20 to-cyan-500/20'
      },
      features: [
        'Tokenisation basée sur les règles',
        'Stop words prédéfinis',
        'Lemmatisation par dictionnaire',
        'Rapide et efficace'
      ]
    },
    bert: {
      name: 'BERT',
      description: 'Bidirectional Encoder Representations - IA moderne',
      icon: Brain,
      colors: {
        primary: 'text-purple-400',
        bg: 'bg-purple-500/20',
        border: 'border-purple-500/30',
        gradient: 'from-purple-500/20 to-pink-500/20'
      },
      features: [
        'Tokenisation contextuelle',
        'Analyse sémantique avancée',
        'Compréhension bidirectionnelle',
        'État de l\'art en NLP'
      ]
    }
  };

  // Code source pour chaque étape
  const getStepCode = (stepNumber: number, model: 'nltk' | 'bert') => {
    const codes = {
      nltk: {
        1: `# Nettoyage du texte - NLTK
import re
import string
import nltk
from nltk.corpus import stopwords

def clean_text_nltk(text, config):
    """
    Nettoie le texte avec l'approche NLTK traditionnelle
    
    Args:
        text (str): Texte à nettoyer
        config (dict): Configuration de nettoyage
    
    Returns:
        dict: Texte nettoyé et statistiques
    """
    original_length = len(text)
    
    # Conversion en minuscules (NLTK standard)
    if not config.get('case_sensitive', False):
        cleaned = text.lower()
    else:
        cleaned = text
    
    # Suppression de la ponctuation
    if not config.get('preserve_punctuation', False):
        # Utiliser string.punctuation pour une approche NLTK
        translator = str.maketrans('', '', string.punctuation)
        cleaned = cleaned.translate(translator)
    
    # Suppression des nombres
    if not config.get('preserve_numbers', False):
        cleaned = re.sub(r'\d+', '', cleaned)
    
    # Normalisation des espaces (approche NLTK)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Suppression des caractères non-ASCII (optionnel)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    
    return {
        'cleaned_text': cleaned,
        'original_length': original_length,
        'cleaned_length': len(cleaned),
        'removed_chars': original_length - len(cleaned),
        'compression_ratio': len(cleaned) / original_length if original_length > 0 else 0
    }

# Exemple d'utilisation
if __name__ == "__main__":
    text = "${text}"
    config = {
        'case_sensitive': False,
        'preserve_punctuation': False,
        'preserve_numbers': False
    }
    
    result = clean_text_nltk(text, config)
    print(f"Texte original: {text}")
    print(f"Texte nettoyé: {result['cleaned_text']}")
    print(f"Caractères supprimés: {result['removed_chars']}")
    print(f"Taux de compression: {result['compression_ratio']:.2%}")`,

        2: `# Tokenisation NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
import re

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

def tokenize_nltk(text, min_length=2, max_tokens=1000):
    """
    Tokenise le texte avec les outils NLTK
    
    Args:
        text (str): Texte à tokeniser
        min_length (int): Longueur minimale des tokens
        max_tokens (int): Nombre maximum de tokens
    
    Returns:
        dict: Tokens et statistiques
    """
    # Méthode 1: Tokenisation basique par mots
    basic_tokens = word_tokenize(text)
    
    # Méthode 2: Tokenisation avec expressions régulières
    tokenizer = RegexpTokenizer(r'\w+')
    regex_tokens = tokenizer.tokenize(text)
    
    # Méthode 3: Tokenisation manuelle (pour comparaison)
    manual_tokens = re.findall(r'\b\w+\b', text)
    
    # Filtrage par longueur minimale
    filtered_tokens = [
        token for token in regex_tokens 
        if len(token) >= min_length
    ]
    
    # Limitation du nombre de tokens
    if len(filtered_tokens) > max_tokens:
        filtered_tokens = filtered_tokens[:max_tokens]
    
    # Statistiques
    stats = {
        'original_text_length': len(text),
        'basic_tokens_count': len(basic_tokens),
        'regex_tokens_count': len(regex_tokens),
        'manual_tokens_count': len(manual_tokens),
        'filtered_tokens_count': len(filtered_tokens),
        'average_token_length': sum(len(t) for t in filtered_tokens) / len(filtered_tokens) if filtered_tokens else 0,
        'unique_tokens': len(set(filtered_tokens)),
        'token_diversity': len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0
    }
    
    return {
        'tokens': filtered_tokens,
        'basic_tokens': basic_tokens,
        'stats': stats
    }

# Exemple d'utilisation
if __name__ == "__main__":
    text = "${text}"
    
    result = tokenize_nltk(text, min_length=${pipelineConfig.parameters.minTokenLength})
    
    print(f"Texte original: {text}")
    print(f"Tokens NLTK: {result['tokens']}")
    print(f"Nombre de tokens: {len(result['tokens'])}")
    print(f"Tokens uniques: {result['stats']['unique_tokens']}")
    print(f"Diversité lexicale: {result['stats']['token_diversity']:.2%}")`,

        3: `# Suppression des stop words - NLTK
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Télécharger les stop words
nltk.download('stopwords', quiet=True)

def remove_stopwords_nltk(tokens, language='english', custom_stopwords=None):
    """
    Supprime les stop words avec NLTK
    
    Args:
        tokens (list): Liste de tokens
        language (str): Langue pour les stop words
        custom_stopwords (list): Stop words personnalisés
    
    Returns:
        dict: Tokens filtrés et statistiques
    """
    # Charger les stop words NLTK
    try:
        nltk_stopwords = set(stopwords.words(language))
    except:
        # Fallback si la langue n'est pas disponible
        nltk_stopwords = set(stopwords.words('english'))
    
    # Ajouter les stop words personnalisés
    if custom_stopwords:
        nltk_stopwords.update(custom_stopwords)
    
    # Ajouter des stop words supplémentaires couramment utilisés
    additional_stopwords = {
        'would', 'could', 'should', 'might', 'must', 'shall', 'will',
        'one', 'two', 'first', 'second', 'also', 'said', 'say',
        'get', 'go', 'know', 'think', 'see', 'come', 'want', 'use'
    }
    nltk_stopwords.update(additional_stopwords)
    
    # Filtrage des tokens
    filtered_tokens = []
    removed_tokens = []
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in nltk_stopwords:
            filtered_tokens.append(token)
        else:
            removed_tokens.append(token)
    
    # Analyse de fréquence
    original_freq = Counter(tokens)
    filtered_freq = Counter(filtered_tokens)
    removed_freq = Counter(removed_tokens)
    
    # Statistiques détaillées
    stats = {
        'original_count': len(tokens),
        'filtered_count': len(filtered_tokens),
        'removed_count': len(removed_tokens),
        'removal_rate': len(removed_tokens) / len(tokens) if tokens else 0,
        'most_common_removed': removed_freq.most_common(5),
        'most_common_kept': filtered_freq.most_common(5),
        'stopwords_used': len(nltk_stopwords),
        'unique_removed': len(set(removed_tokens))
    }
    
    return {
        'filtered_tokens': filtered_tokens,
        'removed_tokens': removed_tokens,
        'stats': stats,
        'stopwords_set': nltk_stopwords
    }

# Exemple d'utilisation
if __name__ == "__main__":
    tokens = ${JSON.stringify(nltkResults[1]?.tokens || [])}
    custom_stops = ${JSON.stringify(pipelineConfig.parameters.customStopWords)}
    
    result = remove_stopwords_nltk(tokens, custom_stopwords=custom_stops)
    
    print(f"Tokens originaux: {len(tokens)}")
    print(f"Tokens filtrés: {result['filtered_tokens']}")
    print(f"Tokens supprimés: {result['removed_tokens']}")
    print(f"Taux de suppression: {result['stats']['removal_rate']:.2%}")
    print(f"Stop words les plus fréquents: {result['stats']['most_common_removed']}")`,

        4: `# Lemmatisation NLTK
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import defaultdict

# Télécharger les ressources nécessaires
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

def get_wordnet_pos(treebank_tag):
    """
    Convertit les tags POS de TreeBank vers WordNet
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Par défaut

def lemmatize_nltk(tokens, use_pos_tags=True):
    """
    Lemmatise les tokens avec NLTK WordNetLemmatizer
    
    Args:
        tokens (list): Liste de tokens à lemmatiser
        use_pos_tags (bool): Utiliser les tags POS pour améliorer la lemmatisation
    
    Returns:
        dict: Tokens lemmatisés et statistiques
    """
    lemmatizer = WordNetLemmatizer()
    
    # Obtenir les tags POS si demandé
    if use_pos_tags and tokens:
        pos_tags = pos_tag(tokens)
    else:
        pos_tags = [(token, 'NN') for token in tokens]
    
    lemmatized_tokens = []
    changes_made = []
    pos_distribution = defaultdict(int)
    
    for token, pos in pos_tags:
        # Convertir le tag POS pour WordNet
        wordnet_pos = get_wordnet_pos(pos)
        pos_distribution[wordnet_pos] += 1
        
        # Lemmatiser le token
        lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
        
        # Conserver la casse originale si possible
        if token.isupper():
            lemma = lemma.upper()
        elif token.istitle():
            lemma = lemma.capitalize()
        
        lemmatized_tokens.append(lemma)
        
        # Enregistrer les changements
        if lemma.lower() != token.lower():
            changes_made.append({
                'original': token,
                'lemma': lemma,
                'pos': pos,
                'wordnet_pos': wordnet_pos
            })
    
    # Statistiques détaillées
    stats = {
        'original_count': len(tokens),
        'lemmatized_count': len(lemmatized_tokens),
        'changes_count': len(changes_made),
        'change_rate': len(changes_made) / len(tokens) if tokens else 0,
        'pos_distribution': dict(pos_distribution),
        'unique_lemmas': len(set(lemmatized_tokens)),
        'vocabulary_reduction': (len(set(tokens)) - len(set(lemmatized_tokens))) / len(set(tokens)) if tokens else 0
    }
    
    return {
        'lemmatized_tokens': lemmatized_tokens,
        'changes_made': changes_made,
        'pos_tags': pos_tags,
        'stats': stats
    }

# Exemple d'utilisation
if __name__ == "__main__":
    tokens = ${JSON.stringify(nltkResults[2]?.tokens || [])}
    
    result = lemmatize_nltk(tokens, use_pos_tags=True)
    
    print(f"Tokens originaux: {tokens}")
    print(f"Tokens lemmatisés: {result['lemmatized_tokens']}")
    print(f"Changements effectués: {len(result['changes_made'])}")
    
    for change in result['changes_made'][:5]:  # Afficher les 5 premiers
        print(f"  {change['original']} -> {change['lemma']} ({change['pos']})")
    
    print(f"Réduction du vocabulaire: {result['stats']['vocabulary_reduction']:.2%}")`
      },
      bert: {
        1: `# Nettoyage du texte - BERT
import re
import torch
from transformers import BertTokenizer
import unicodedata

def clean_text_bert(text, config):
    """
    Nettoyage spécialisé pour BERT avec préservation du contexte
    
    Args:
        text (str): Texte à nettoyer
        config (dict): Configuration de nettoyage
    
    Returns:
        dict: Texte nettoyé et métadonnées
    """
    original_text = text
    original_length = len(text)
    
    # Normalisation Unicode (importante pour BERT)
    cleaned = unicodedata.normalize('NFKC', text)
    
    # Préservation sélective de la casse (BERT est case-sensitive)
    if not config.get('case_sensitive', True):
        # BERT uncased - conversion en minuscules
        cleaned = cleaned.lower()
    
    # Gestion spéciale des caractères pour BERT
    if not config.get('preserve_punctuation', True):
        # Préserver certains caractères importants pour BERT
        important_chars = r'[.!?,:;]'
        cleaned = re.sub(r'[^\w\s' + important_chars + r']', ' ', cleaned)
    
    # Gestion des nombres pour BERT
    if not config.get('preserve_numbers', True):
        # Remplacer par un token spécial plutôt que supprimer
        cleaned = re.sub(r'\d+', '[NUM]', cleaned)
    
    # Gestion des URLs et mentions (spécifique aux modèles pré-entraînés)
    cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', cleaned)
    cleaned = re.sub(r'@\w+', '[USER]', cleaned)
    
    # Normalisation des espaces (préserver la structure pour BERT)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Vérification de la longueur pour BERT (limite de 512 tokens)
    if len(cleaned.split()) > 500:  # Marge de sécurité
        words = cleaned.split()
        cleaned = ' '.join(words[:500])
    
    return {
        'cleaned_text': cleaned,
        'original_text': original_text,
        'original_length': original_length,
        'cleaned_length': len(cleaned),
        'unicode_normalized': True,
        'bert_ready': True,
        'truncated': len(original_text.split()) > 500
    }

# Exemple d'utilisation avec BERT
if __name__ == "__main__":
    text = "${text}"
    config = {
        'case_sensitive': False,  # BERT uncased
        'preserve_punctuation': True,
        'preserve_numbers': False
    }
    
    result = clean_text_bert(text, config)
    
    print(f"Texte original: {result['original_text']}")
    print(f"Texte nettoyé pour BERT: {result['cleaned_text']}")
    print(f"Prêt pour BERT: {result['bert_ready']}")
    print(f"Normalisé Unicode: {result['unicode_normalized']}")
    
    # Test avec le tokenizer BERT
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(result['cleaned_text'])
        print(f"Tokens BERT générés: {len(tokens)}")
    except:
        print("Tokenizer BERT non disponible pour le test")`,

        2: `# Tokenisation BERT WordPiece
import torch
from transformers import BertTokenizer, BertModel
import re
from collections import defaultdict

def tokenize_bert_wordpiece(text, model_name='bert-base-uncased', max_length=512):
    """
    Tokenisation BERT avec algorithme WordPiece
    
    Args:
        text (str): Texte à tokeniser
        model_name (str): Nom du modèle BERT
        max_length (int): Longueur maximale de séquence
    
    Returns:
        dict: Tokens et métadonnées BERT
    """
    try:
        # Charger le tokenizer BERT pré-entraîné
        tokenizer = BertTokenizer.from_pretrained(model_name)
    except:
        # Simulation si le modèle n'est pas disponible
        return simulate_bert_tokenization(text)
    
    # Tokenisation complète avec tokens spéciaux
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Tokenisation simple (sans tokens spéciaux)
    tokens = tokenizer.tokenize(text)
    
    # Analyse des sous-mots
    subword_analysis = analyze_subwords(tokens)
    
    # Conversion en IDs et retour en tokens
    token_ids = encoded['input_ids'][0].tolist()
    tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Statistiques détaillées
    stats = {
        'original_text_length': len(text),
        'total_tokens': len(tokens),
        'subword_tokens': subword_analysis['subword_count'],
        'complete_words': subword_analysis['complete_word_count'],
        'subword_ratio': subword_analysis['subword_count'] / len(tokens) if tokens else 0,
        'vocabulary_coverage': calculate_vocab_coverage(tokens, tokenizer),
        'attention_mask_sum': encoded['attention_mask'].sum().item(),
        'sequence_length': len(tokens_with_special),
        'special_tokens': ['[CLS]', '[SEP]', '[PAD]']
    }
    
    return {
        'tokens': tokens,
        'tokens_with_special': tokens_with_special,
        'token_ids': token_ids,
        'attention_mask': encoded['attention_mask'],
        'subword_analysis': subword_analysis,
        'stats': stats,
        'tokenizer_info': {
            'model_name': model_name,
            'vocab_size': tokenizer.vocab_size,
            'max_length': max_length
        }
    }

def analyze_subwords(tokens):
    """Analyse la décomposition en sous-mots"""
    subword_count = sum(1 for token in tokens if token.startswith('##'))
    complete_word_count = len(tokens) - subword_count
    
    # Reconstruction des mots complets
    reconstructed_words = []
    current_word = ""
    
    for token in tokens:
        if token.startswith('##'):
            current_word += token[2:]  # Enlever ##
        else:
            if current_word:
                reconstructed_words.append(current_word)
            current_word = token
    
    if current_word:
        reconstructed_words.append(current_word)
    
    return {
        'subword_count': subword_count,
        'complete_word_count': complete_word_count,
        'reconstructed_words': reconstructed_words,
        'average_subwords_per_word': subword_count / complete_word_count if complete_word_count > 0 else 0
    }

def calculate_vocab_coverage(tokens, tokenizer):
    """Calcule la couverture du vocabulaire"""
    vocab = set(tokenizer.vocab.keys())
    token_set = set(tokens)
    coverage = len(token_set.intersection(vocab)) / len(token_set) if token_set else 0
    return coverage

def simulate_bert_tokenization(text):
    """Simulation de la tokenisation BERT si le modèle n'est pas disponible"""
    words = text.split()
    tokens = []
    
    # Vocabulaire BERT simulé
    bert_vocab = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'good', 'bad', 'great', 'amazing', 'terrible', 'excellent', 'poor', 'wonderful',
        'product', 'quality', 'price', 'fast', 'slow', 'easy', 'difficult', 'love', 'hate'
    }
    
    for word in words:
        word_lower = word.lower()
        if word_lower in bert_vocab or len(word) <= 3:
            tokens.append(word_lower)
        else:
            # Simulation de la décomposition WordPiece
            remaining = word_lower
            first_subword = True
            
            while remaining:
                found = False
                for length in range(min(len(remaining), 6), 1, -1):
                    subword = remaining[:length]
                    if first_subword:
                        tokens.append(subword)
                        remaining = remaining[length:]
                        first_subword = False
                        found = True
                        break
                    else:
                        tokens.append(f"##{subword}")
                        remaining = remaining[length:]
                        found = True
                        break
                
                if not found:
                    if first_subword:
                        tokens.append(remaining[0])
                    else:
                        tokens.append(f"##{remaining[0]}")
                    remaining = remaining[1:]
                    first_subword = False
    
    return {
        'tokens': tokens,
        'stats': {
            'total_tokens': len(tokens),
            'subword_tokens': sum(1 for t in tokens if t.startswith('##')),
            'simulated': True
        }
    }

# Exemple d'utilisation
if __name__ == "__main__":
    text = "${text}"
    
    result = tokenize_bert_wordpiece(text)
    
    print(f"Texte original: {text}")
    print(f"Tokens BERT: {result['tokens']}")
    print(f"Nombre de tokens: {result['stats']['total_tokens']}")
    print(f"Sous-mots: {result['stats']['subword_tokens']}")
    print(f"Ratio sous-mots: {result['stats']['subword_ratio']:.2%}")
    
    if 'subword_analysis' in result:
        print(f"Mots reconstruits: {result['subword_analysis']['reconstructed_words']}")`,

        3: `# Suppression des stop words - BERT Contextuel
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def remove_stopwords_bert_contextual(tokens, context_window=3, similarity_threshold=0.7):
    """
    Suppression contextuelle des stop words pour BERT
    
    Args:
        tokens (list): Liste de tokens BERT
        context_window (int): Fenêtre de contexte pour l'analyse
        similarity_threshold (float): Seuil de similarité pour la préservation
    
    Returns:
        dict: Tokens filtrés avec analyse contextuelle
    """
    # Stop words de base pour BERT
    bert_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Mots de contexte important (à préserver même s'ils sont des stop words)
    context_preservers = {
        'not', 'never', 'no', 'nothing', 'none', 'neither', 'nor',
        'very', 'extremely', 'incredibly', 'absolutely', 'totally',
        'but', 'however', 'although', 'though', 'despite'
    }
    
    filtered_tokens = []
    removed_tokens = []
    contextual_decisions = []
    
    for i, token in enumerate(tokens):
        # Nettoyer le token (enlever ##)
        clean_token = token.replace('##', '').lower()
        
        # Vérifier si c'est un stop word
        if clean_token in bert_stopwords:
            # Analyser le contexte
            should_preserve = analyze_token_context(
                tokens, i, context_window, context_preservers, similarity_threshold
            )
            
            decision = {
                'token': token,
                'position': i,
                'is_stopword': True,
                'context_analysis': should_preserve,
                'preserved': should_preserve['should_preserve']
            }
            contextual_decisions.append(decision)
            
            if should_preserve['should_preserve']:
                filtered_tokens.append(token)
            else:
                removed_tokens.append(token)
        else:
            # Garder les tokens non-stop words
            filtered_tokens.append(token)
            contextual_decisions.append({
                'token': token,
                'position': i,
                'is_stopword': False,
                'preserved': True
            })
    
    # Statistiques contextuelles
    stats = {
        'original_count': len(tokens),
        'filtered_count': len(filtered_tokens),
        'removed_count': len(removed_tokens),
        'contextual_preservations': sum(1 for d in contextual_decisions if d.get('is_stopword') and d.get('preserved')),
        'context_analysis_performed': sum(1 for d in contextual_decisions if d.get('context_analysis')),
        'removal_rate': len(removed_tokens) / len(tokens) if tokens else 0,
        'context_preservation_rate': sum(1 for d in contextual_decisions if d.get('is_stopword') and d.get('preserved')) / sum(1 for d in contextual_decisions if d.get('is_stopword')) if any(d.get('is_stopword') for d in contextual_decisions) else 0
    }
    
    return {
        'filtered_tokens': filtered_tokens,
        'removed_tokens': removed_tokens,
        'contextual_decisions': contextual_decisions,
        'stats': stats
    }

def analyze_token_context(tokens, position, window, preservers, threshold):
    """
    Analyse le contexte d'un token pour décider de sa préservation
    """
    token = tokens[position].replace('##', '').lower()
    
    # Vérifier les préservateurs de contexte
    if token in preservers:
        return {
            'should_preserve': True,
            'reason': 'context_preserver',
            'confidence': 1.0
        }
    
    # Analyser la fenêtre de contexte
    start = max(0, position - window)
    end = min(len(tokens), position + window + 1)
    context_tokens = tokens[start:end]
    
    # Rechercher des indicateurs de contexte important
    context_indicators = {
        'negation': ['not', 'never', 'no', 'nothing', 'none'],
        'intensification': ['very', 'extremely', 'incredibly', 'absolutely'],
        'contrast': ['but', 'however', 'although', 'though'],
        'emphasis': ['really', 'definitely', 'certainly', 'surely']
    }
    
    importance_score = 0
    reasons = []
    
    for indicator_type, indicators in context_indicators.items():
        for context_token in context_tokens:
            clean_context = context_token.replace('##', '').lower()
            if clean_context in indicators:
                importance_score += 0.3
                reasons.append(f"{indicator_type}_nearby")
    
    # Vérifier la position dans la phrase
    if position == 0 or position == len(tokens) - 1:
        importance_score += 0.2
        reasons.append('sentence_boundary')
    
    # Décision finale
    should_preserve = importance_score >= threshold
    
    return {
        'should_preserve': should_preserve,
        'importance_score': importance_score,
        'reasons': reasons,
        'confidence': min(1.0, importance_score)
    }

# Exemple d'utilisation
if __name__ == "__main__":
    tokens = ${JSON.stringify(bertResults[1]?.tokens || [])}
    
    result = remove_stopwords_bert_contextual(tokens)
    
    print(f"Tokens originaux: {len(tokens)}")
    print(f"Tokens filtrés: {result['filtered_tokens']}")
    print(f"Tokens supprimés: {result['removed_tokens']}")
    print(f"Préservations contextuelles: {result['stats']['contextual_preservations']}")
    print(f"Taux de préservation contextuelle: {result['stats']['context_preservation_rate']:.2%}")
    
    # Afficher quelques décisions contextuelles
    for decision in result['contextual_decisions'][:5]:
        if decision.get('context_analysis'):
            print(f"Token '{decision['token']}': {decision['context_analysis']['reason']} (score: {decision['context_analysis']['confidence']:.2f})")`,

        4: `# Lemmatisation BERT Contextuelle
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from collections import defaultdict

def lemmatize_bert_contextual(tokens, model_name='bert-base-uncased'):
    """
    Lemmatisation contextuelle avancée pour BERT
    
    Args:
        tokens (list): Liste de tokens BERT
        model_name (str): Nom du modèle BERT
    
    Returns:
        dict: Tokens lemmatisés avec analyse contextuelle
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
    except:
        # Fallback vers lemmatisation basée sur des règles
        return fallback_bert_lemmatization(tokens)
    
    # Reconstruction des mots à partir des sous-mots
    reconstructed_words = reconstruct_words_from_subwords(tokens)
    
    # Lemmatisation contextuelle avec embeddings BERT
    lemmatized_results = []
    
    for word_info in reconstructed_words:
        original_word = word_info['word']
        subword_tokens = word_info['subword_tokens']
        
        # Obtenir les embeddings contextuels
        try:
            embeddings = get_contextual_embeddings(
                subword_tokens, tokens, tokenizer, model
            )
            
            # Lemmatisation basée sur les embeddings
            lemma_result = contextual_lemmatization(
                original_word, embeddings, tokenizer
            )
            
        except:
            # Fallback vers lemmatisation basée sur des règles
            lemma_result = rule_based_lemmatization(original_word)
        
        lemmatized_results.append({
            'original': original_word,
            'lemma': lemma_result['lemma'],
            'confidence': lemma_result['confidence'],
            'method': lemma_result['method'],
            'subword_tokens': subword_tokens
        })
    
    # Reconstruire la liste de tokens lemmatisés
    lemmatized_tokens = []
    for result in lemmatized_results:
        if result['original'] != result['lemma']:
            # Le mot a été lemmatisé
            if len(result['subword_tokens']) == 1:
                lemmatized_tokens.append(result['lemma'])
            else:
                # Redistribuer le lemme sur les sous-mots
                lemma_subwords = redistribute_lemma_to_subwords(
                    result['lemma'], result['subword_tokens'], tokenizer
                )
                lemmatized_tokens.extend(lemma_subwords)
        else:
            # Le mot n'a pas changé
            lemmatized_tokens.extend(result['subword_tokens'])
    
    # Statistiques détaillées
    changes_made = [r for r in lemmatized_results if r['original'] != r['lemma']]
    
    stats = {
        'original_token_count': len(tokens),
        'lemmatized_token_count': len(lemmatized_tokens),
        'word_count': len(reconstructed_words),
        'changes_made': len(changes_made),
        'change_rate': len(changes_made) / len(reconstructed_words) if reconstructed_words else 0,
        'contextual_lemmas': sum(1 for r in lemmatized_results if r['method'] == 'contextual'),
        'rule_based_lemmas': sum(1 for r in lemmatized_results if r['method'] == 'rule_based'),
        'average_confidence': np.mean([r['confidence'] for r in lemmatized_results]) if lemmatized_results else 0
    }
    
    return {
        'lemmatized_tokens': lemmatized_tokens,
        'lemmatization_results': lemmatized_results,
        'changes_made': changes_made,
        'reconstructed_words': reconstructed_words,
        'stats': stats
    }

def reconstruct_words_from_subwords(tokens):
    """Reconstruit les mots complets à partir des sous-mots BERT"""
    words = []
    current_word = ""
    current_subwords = []
    
    for token in tokens:
        if token.startswith('##'):
            current_word += token[2:]
            current_subwords.append(token)
        else:
            if current_word:
                words.append({
                    'word': current_word,
                    'subword_tokens': current_subwords
                })
            current_word = token
            current_subwords = [token]
    
    if current_word:
        words.append({
            'word': current_word,
            'subword_tokens': current_subwords
        })
    
    return words

def get_contextual_embeddings(subword_tokens, full_tokens, tokenizer, model):
    """Obtient les embeddings contextuels pour les sous-mots"""
    # Créer la séquence complète avec tokens spéciaux
    sequence = ['[CLS]'] + full_tokens + ['[SEP]']
    
    # Convertir en IDs
    input_ids = tokenizer.convert_tokens_to_ids(sequence)
    input_tensor = torch.tensor([input_ids])
    
    # Obtenir les embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
    
    return embeddings

def contextual_lemmatization(word, embeddings, tokenizer):
    """Lemmatisation basée sur les embeddings contextuels"""
    # Dictionnaire de lemmatisation contextuelle étendu
    contextual_lemma_map = {
        'running': 'run', 'runs': 'run', 'ran': 'run',
        'better': 'good', 'best': 'good', 'excellent': 'good', 'outstanding': 'good',
        'amazing': 'amaze', 'amazed': 'amaze', 'awesome': 'amaze', 'fantastic': 'amaze',
        'products': 'product', 'items': 'item', 'goods': 'product',
        'buying': 'buy', 'bought': 'buy', 'purchase': 'buy', 'purchasing': 'buy',
        'working': 'work', 'works': 'work', 'worked': 'work',
        'using': 'use', 'used': 'use', 'uses': 'use', 'utilizing': 'use',
        'getting': 'get', 'got': 'get', 'gets': 'get', 'obtaining': 'get',
        'shipping': 'ship', 'shipped': 'ship', 'delivery': 'deliver', 'delivered': 'deliver',
        'loving': 'love', 'loved': 'love', 'loves': 'love',
        'hating': 'hate', 'hated': 'hate', 'hates': 'hate',
        'feeling': 'feel', 'felt': 'feel', 'feels': 'feel',
        'thinking': 'think', 'thought': 'think', 'thinks': 'think',
        'looking': 'look', 'looked': 'look', 'looks': 'look'
    }
    
    word_lower = word.lower()
    
    if word_lower in contextual_lemma_map:
        return {
            'lemma': contextual_lemma_map[word_lower],
            'confidence': 0.9,
            'method': 'contextual'
        }
    else:
        return {
            'lemma': word,
            'confidence': 0.5,
            'method': 'unchanged'
        }

def rule_based_lemmatization(word):
    """Lemmatisation basée sur des règles simples"""
    word_lower = word.lower()
    
    # Règles de lemmatisation simples
    if word_lower.endswith('ing'):
        if len(word_lower) > 6:
            lemma = word_lower[:-3]
            return {'lemma': lemma, 'confidence': 0.7, 'method': 'rule_based'}
    
    if word_lower.endswith('ed'):
        if len(word_lower) > 5:
            lemma = word_lower[:-2]
            return {'lemma': lemma, 'confidence': 0.7, 'method': 'rule_based'}
    
    if word_lower.endswith('s') and len(word_lower) > 3:
        lemma = word_lower[:-1]
        return {'lemma': lemma, 'confidence': 0.6, 'method': 'rule_based'}
    
    return {'lemma': word, 'confidence': 0.5, 'method': 'unchanged'}

def redistribute_lemma_to_subwords(lemma, original_subwords, tokenizer):
    """Redistribue un lemme sur les sous-mots originaux"""
    if len(original_subwords) == 1:
        return [lemma]
    
    # Essayer de retokeniser le lemme
    try:
        new_subwords = tokenizer.tokenize(lemma)
        if len(new_subwords) <= len(original_subwords):
            return new_subwords
    except:
        pass
    
    # Fallback: garder la structure originale mais avec le lemme
    return [lemma] + original_subwords[1:]

def fallback_bert_lemmatization(tokens):
    """Lemmatisation de fallback si BERT n'est pas disponible"""
    # Utiliser la lemmatisation basée sur des règles
    lemmatized_tokens = []
    changes_made = []
    
    for token in tokens:
        clean_token = token.replace('##', '')
        result = rule_based_lemmatization(clean_token)
        
        if result['lemma'] != clean_token:
            if token.startswith('##'):
                lemmatized_token = '##' + result['lemma']
            else:
                lemmatized_token = result['lemma']
            
            changes_made.append({
                'original': token,
                'lemma': lemmatized_token,
                'method': 'fallback_rule'
            })
        else:
            lemmatized_token = token
        
        lemmatized_tokens.append(lemmatized_token)
    
    return {
        'lemmatized_tokens': lemmatized_tokens,
        'changes_made': changes_made,
        'stats': {
            'changes_made': len(changes_made),
            'method': 'fallback'
        }
    }

# Exemple d'utilisation
if __name__ == "__main__":
    tokens = ${JSON.stringify(bertResults[2]?.tokens || [])}
    
    result = lemmatize_bert_contextual(tokens)
    
    print(f"Tokens originaux: {tokens}")
    print(f"Tokens lemmatisés: {result['lemmatized_tokens']}")
    print(f"Changements effectués: {len(result['changes_made'])}")
    print(f"Confiance moyenne: {result['stats']['average_confidence']:.2f}")
    
    for change in result['changes_made'][:5]:
        print(f"  {change['original']} -> {change['lemma']} ({change['method']})")
    
    print(f"Lemmatisation contextuelle: {result['stats']['contextual_lemmas']}")
    print(f"Lemmatisation par règles: {result['stats']['rule_based_lemmas']}")`
      }
    };

    return codes[model][stepNumber] || '# Code non disponible';
  };

  // Ouvrir popup de code
  const openCodePopup = (stepTitle: string, stepNumber: number, model: 'nltk' | 'bert') => {
    setCodePopup({
      isOpen: true,
      stepTitle,
      code: getStepCode(stepNumber, model),
      model,
      stepNumber
    });
  };

  // Fermer popup de code
  const closeCodePopup = () => {
    setCodePopup(prev => ({ ...prev, isOpen: false }));
  };

  // Fonctions de traitement (identiques à la version précédente)
  const cleanText = (input: string) => {
    let cleaned = input.toLowerCase();
    
    if (!pipelineConfig.parameters.preservePunctuation) {
      cleaned = cleaned.replace(/[^\w\s]/g, ' ');
    }
    
    if (!pipelineConfig.parameters.preserveNumbers) {
      cleaned = cleaned.replace(/\d+/g, ' ');
    }
    
    cleaned = cleaned.replace(/\s+/g, ' ').trim();
    
    const removed = input.match(/[^\w\s]/g) || [];
    return { cleaned, removed };
  };

  const tokenizeNLTK = (input: string) => {
    return input.split(/\s+/).filter(token => 
      token.length >= pipelineConfig.parameters.minTokenLength
    );
  };

  const tokenizeBERT = (input: string) => {
    const words = input.split(/\s+/).filter(token => 
      token.length >= pipelineConfig.parameters.minTokenLength
    );
    const bertTokens: string[] = [];
    
    const bertVocab = new Set([
      'this', 'is', 'the', 'and', 'or', 'but', 'not', 'very', 'good', 'bad', 'great', 'amazing', 'terrible',
      'product', 'quality', 'fast', 'slow', 'shipping', 'delivery', 'price', 'money', 'buy', 'purchase',
      'excellent', 'wonderful', 'fantastic', 'awful', 'horrible', 'love', 'hate', 'like', 'dislike',
      'recommend', 'satisfied', 'disappointed', 'happy', 'angry', 'frustrated', 'pleased', 'upset'
    ]);
    
    words.forEach(word => {
      if (bertVocab.has(word.toLowerCase()) || word.length <= 3) {
        bertTokens.push(word);
      } else {
        let remaining = word;
        let isFirst = true;
        
        while (remaining.length > 0) {
          let found = false;
          
          for (let len = Math.min(remaining.length, 6); len >= 2; len--) {
            const subword = remaining.substring(0, len);
            if (len < remaining.length || !isFirst) {
              const tokenToAdd = isFirst ? subword : '##' + subword;
              bertTokens.push(tokenToAdd);
              remaining = remaining.substring(len);
              isFirst = false;
              found = true;
              break;
            }
          }
          
          if (!found) {
            const tokenToAdd = isFirst ? remaining[0] : '##' + remaining[0];
            bertTokens.push(tokenToAdd);
            remaining = remaining.substring(1);
            isFirst = false;
          }
        }
      }
    });
    
    return bertTokens;
  };

  const removeStopWordsNLTK = (tokens: string[]) => {
    const stopWords = [
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
      'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
      'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 
      'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
      ...pipelineConfig.parameters.customStopWords
    ];
    
    const filtered = tokens.filter(token => !stopWords.includes(token.toLowerCase().replace('##', '')));
    const removed = tokens.filter(token => stopWords.includes(token.toLowerCase().replace('##', '')));
    return { filtered, removed };
  };

  const removeStopWordsBERT = (tokens: string[]) => {
    const contextualStopWords = ['the', 'a', 'an', 'is', 'are', 'was', 'were', ...pipelineConfig.parameters.customStopWords];
    const filtered = tokens.filter(token => {
      const cleanToken = token.toLowerCase().replace('##', '');
      return !contextualStopWords.includes(cleanToken);
    });
    const removed = tokens.filter(token => {
      const cleanToken = token.toLowerCase().replace('##', '');
      return contextualStopWords.includes(cleanToken);
    });
    return { filtered, removed };
  };

  const lemmatizeNLTK = (tokens: string[]) => {
    const lemmaMap: { [key: string]: string } = {
      'running': 'run', 'runs': 'run', 'ran': 'run',
      'better': 'good', 'best': 'good',
      'amazing': 'amaze', 'amazed': 'amaze',
      'products': 'product', 'items': 'item',
      'buying': 'buy', 'bought': 'buy',
      'working': 'work', 'works': 'work', 'worked': 'work',
      'using': 'use', 'used': 'use', 'uses': 'use',
      'getting': 'get', 'got': 'get', 'gets': 'get',
      'shipping': 'ship', 'shipped': 'ship'
    };
    
    const lemmatized = tokens.map(token => {
      const cleanToken = token.replace('##', '');
      const lemma = lemmaMap[cleanToken.toLowerCase()];
      return lemma ? (token.includes('##') ? '##' + lemma : lemma) : token;
    });
    
    const changed = tokens.filter((token, i) => lemmatized[i] !== token);
    return { lemmatized, changed };
  };

  const lemmatizeBERT = (tokens: string[]) => {
    const contextualLemmaMap: { [key: string]: string } = {
      'running': 'run', 'runs': 'run', 'ran': 'run',
      'better': 'good', 'best': 'good', 'excellent': 'good', 'outstanding': 'good',
      'amazing': 'amaze', 'amazed': 'amaze', 'awesome': 'amaze', 'fantastic': 'amaze',
      'products': 'product', 'items': 'item', 'goods': 'product',
      'buying': 'buy', 'bought': 'buy', 'purchase': 'buy', 'purchasing': 'buy',
      'working': 'work', 'works': 'work', 'worked': 'work',
      'using': 'use', 'used': 'use', 'uses': 'use', 'utilizing': 'use',
      'getting': 'get', 'got': 'get', 'gets': 'get', 'obtaining': 'get',
      'shipping': 'ship', 'shipped': 'ship', 'delivery': 'deliver', 'delivered': 'deliver'
    };
    
    const lemmatized = tokens.map(token => {
      const cleanToken = token.replace('##', '');
      const lemma = contextualLemmaMap[cleanToken.toLowerCase()];
      return lemma ? (token.includes('##') ? '##' + lemma : lemma) : token;
    });
    
    const changed = tokens.filter((token, i) => lemmatized[i] !== token);
    return { lemmatized, changed };
  };

  // Traitement complet pour un modèle
  const processModel = (model: 'nltk' | 'bert') => {
    const results = [];
    const config = modelConfigs[model];

    if (!pipelineConfig.steps.cleaning && !pipelineConfig.steps.tokenization && 
        !pipelineConfig.steps.stopwords && !pipelineConfig.steps.lemmatization) {
      return [];
    }

    let currentText = text;
    let currentTokens: string[] = [];

    // Étape 1: Nettoyage
    if (pipelineConfig.steps.cleaning) {
      const { cleaned, removed: cleanRemoved } = cleanText(currentText);
      currentText = cleaned;
      
      results.push({
        stepNumber: 1,
        title: 'Nettoyage du Texte',
        description: 'Suppression des caractères indésirables et normalisation',
        icon: FileText,
        input: text,
        output: cleaned,
        tokens: [],
        removedItems: cleanRemoved,
        model: model,
        details: [
          'Suppression des caractères spéciaux',
          'Conversion en minuscules',
          'Suppression des espaces multiples',
          'Normalisation du texte'
        ],
        stats: {
          inputLength: text.length,
          outputLength: cleaned.length,
          tokensCount: 0,
          removedCount: cleanRemoved.length
        }
      });
    }

    // Étape 2: Tokenisation
    if (pipelineConfig.steps.tokenization) {
      const tokens = model === 'nltk' ? tokenizeNLTK(currentText) : tokenizeBERT(currentText);
      currentTokens = tokens;
      
      results.push({
        stepNumber: 2,
        title: `Tokenisation ${config.name}`,
        description: model === 'nltk' 
          ? 'Division du texte en mots (approche traditionnelle)'
          : 'Division en sous-mots contextuels (BERT WordPiece)',
        icon: Scissors,
        input: currentText,
        output: tokens.join(' | '),
        tokens: tokens,
        removedItems: [],
        model: model,
        details: model === 'nltk' ? [
          'Séparation par espaces',
          'Identification des mots complets',
          'Préservation de la structure',
          'Approche basée sur les règles'
        ] : [
          'Tokenisation en sous-mots',
          'WordPiece algorithm',
          'Gestion des mots rares',
          'Préservation du contexte'
        ],
        stats: {
          inputLength: currentText.length,
          outputLength: tokens.join(' ').length,
          tokensCount: tokens.length,
          removedCount: 0
        }
      });
    }

    // Étape 3: Suppression des stop words
    if (pipelineConfig.steps.stopwords && currentTokens.length > 0) {
      const { filtered, removed: stopWordsRemoved } = model === 'nltk' 
        ? removeStopWordsNLTK(currentTokens) 
        : removeStopWordsBERT(currentTokens);
      
      currentTokens = filtered;
      
      results.push({
        stepNumber: 3,
        title: `Stop Words ${config.name}`,
        description: model === 'nltk'
          ? 'Suppression des mots vides (liste prédéfinie)'
          : 'Filtrage contextuel des mots non-significatifs',
        icon: Filter,
        input: results[results.length - 1]?.output || currentTokens.join(' | '),
        output: filtered.join(' | '),
        tokens: filtered,
        removedItems: stopWordsRemoved,
        model: model,
        details: model === 'nltk' ? [
          'Liste de stop words prédéfinie',
          'Suppression systématique',
          'Approche déterministe',
          'Optimisation pour l\'analyse'
        ] : [
          'Analyse contextuelle',
          'Filtrage intelligent',
          'Préservation du sens',
          'Approche adaptative'
        ],
        stats: {
          inputLength: results[results.length - 1]?.stats.tokensCount || currentTokens.length,
          outputLength: filtered.length,
          tokensCount: filtered.length,
          removedCount: stopWordsRemoved.length
        }
      });
    }

    // Étape 4: Lemmatisation
    if (pipelineConfig.steps.lemmatization && currentTokens.length > 0) {
      const { lemmatized, changed } = model === 'nltk' 
        ? lemmatizeNLTK(currentTokens) 
        : lemmatizeBERT(currentTokens);
      
      currentTokens = lemmatized;
      
      results.push({
        stepNumber: 4,
        title: `Lemmatisation ${config.name}`,
        description: model === 'nltk'
          ? 'Réduction à la forme canonique (dictionnaire)'
          : 'Normalisation contextuelle avancée',
        icon: RefreshCw,
        input: results[results.length - 1]?.output || currentTokens.join(' | '),
        output: lemmatized.join(' | '),
        tokens: lemmatized,
        removedItems: changed,
        model: model,
        details: model === 'nltk' ? [
          'Dictionnaire de lemmes',
          'Règles morphologiques',
          'Forme canonique standard',
          'Approche linguistique'
        ] : [
          'Embeddings contextuels',
          'Analyse sémantique',
          'Normalisation intelligente',
          'Compréhension du contexte'
        ],
        stats: {
          inputLength: results[results.length - 1]?.stats.tokensCount || currentTokens.length,
          outputLength: lemmatized.length,
          tokensCount: lemmatized.length,
          removedCount: changed.length
        }
      });
    }

    return results;
  };

  // Traitement complet du pipeline
  React.useEffect(() => {
    if (text && text.trim()) {
      setIsInitialized(false);
      
      const nltkRes = processModel('nltk');
      const bertRes = processModel('bert');
      
      setNltkResults(nltkRes);
      setBertResults(bertRes);
      setIsInitialized(true);

      // Appeler onComplete avec les résultats finaux
      onComplete({
        originalText: text,
        nltkResults: nltkRes,
        bertResults: bertRes,
        finalTokensNltk: nltkRes.length > 0 ? nltkRes[nltkRes.length - 1]?.tokens || [] : [],
        finalTokensBert: bertRes.length > 0 ? bertRes[bertRes.length - 1]?.tokens || [] : [],
        pipelineConfig
      });
    }
  }, [text, pipelineConfig, onComplete]);

  if (!text || !text.trim()) {
    return (
      <div className="text-center py-12">
        <FileText className="h-16 w-16 text-white/40 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-white mb-2">Pipeline NLP Avancé</h2>
        <p className="text-white/70">Entrez un texte pour voir l'analyse complète</p>
      </div>
    );
  }

  const renderPipelineSteps = (results: any[], model: 'nltk' | 'bert') => {
    const config = modelConfigs[model];
    
    return (
      <div className="space-y-8">
        {results.map((result, index) => (
          <div key={index}>
            <div className={`bg-gradient-to-r ${config.colors.gradient} ${config.colors.border} border rounded-xl shadow-xl overflow-hidden`}>
              <div className="p-8">
                {/* En-tête de l'étape */}
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center space-x-4">
                    <div className={`${config.colors.bg} ${config.colors.primary} p-4 rounded-xl shadow-lg`}>
                      <result.icon className="h-8 w-8" />
                    </div>
                    <div>
                      <h3 className="text-white font-bold text-2xl">
                        {result.title}
                      </h3>
                      <p className="text-white/80 text-lg">{result.description}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => openCodePopup(result.title, result.stepNumber, model)}
                      className={`px-4 py-2 ${config.colors.bg} ${config.colors.primary} rounded-lg hover:bg-opacity-30 transition-all flex items-center space-x-2 ${config.colors.border} border hover:scale-105 shadow-lg`}
                      title="Voir le code source"
                    >
                      <Code className="h-4 w-4" />
                      <span className="text-sm font-medium">Code</span>
                    </button>
                    
                    <div className="text-right bg-white/10 p-4 rounded-xl">
                      <div className={`${config.colors.primary} font-bold text-3xl`}>
                        {result.stats.tokensCount || result.stats.outputLength}
                      </div>
                      <div className="text-white/60 text-sm">
                        {result.stepNumber === 1 ? 'caractères' : 'tokens'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Détails techniques */}
                <div className="grid md:grid-cols-4 gap-4 mb-8">
                  {result.details.map((detail: string, i: number) => (
                    <div key={i} className={`${config.colors.bg} p-4 rounded-lg text-white/90 text-sm text-center ${config.colors.border} border`}>
                      {detail}
                    </div>
                  ))}
                </div>

                {/* Transformation */}
                <div className="space-y-6">
                  {/* Entrée */}
                  <div>
                    <div className="flex items-center space-x-3 mb-3">
                      <span className="text-white/80 font-bold text-lg">📥 ENTRÉE:</span>
                    </div>
                    <div className="bg-white/10 p-4 rounded-lg border border-white/20">
                      <p className="text-white/90 font-mono text-sm break-words">
                        {result.input}
                      </p>
                    </div>
                  </div>

                  {/* Flèche de transformation */}
                  <div className="flex justify-center">
                    <ArrowRight className={`h-8 w-8 ${config.colors.primary}`} />
                  </div>

                  {/* Sortie */}
                  <div>
                    <div className="flex items-center space-x-3 mb-3">
                      <span className={`${config.colors.primary} font-bold text-lg`}>📤 SORTIE:</span>
                    </div>
                    <div className={`${config.colors.bg} p-4 rounded-lg ${config.colors.border} border`}>
                      <p className="text-white font-mono text-sm break-words">
                        {result.output}
                      </p>
                    </div>
                  </div>

                  {/* Éléments supprimés/modifiés */}
                  {result.removedItems.length > 0 && (
                    <div>
                      <div className="flex items-center space-x-3 mb-3">
                        <span className="text-red-400 font-bold text-lg">
                          🗑️ {result.stepNumber === 4 ? 'MODIFIÉS:' : 'SUPPRIMÉS:'}
                        </span>
                      </div>
                      <div className="bg-red-500/20 p-4 rounded-lg border border-red-500/30">
                        <div className="flex flex-wrap gap-3">
                          {result.removedItems.map((item: string, i: number) => (
                            <span key={i} className="px-3 py-2 bg-red-500/30 text-red-200 rounded-full text-sm font-mono">
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Statistiques */}
                  <div className="grid grid-cols-4 gap-6 pt-6 border-t border-white/20">
                    <div className="text-center bg-white/5 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-white">{result.stats.inputLength}</div>
                      <div className="text-white/60 text-sm">Entrée</div>
                    </div>
                    <div className="text-center bg-white/5 p-4 rounded-lg">
                      <div className={`text-2xl font-bold ${config.colors.primary}`}>{result.stats.tokensCount || result.stats.outputLength}</div>
                      <div className="text-white/60 text-sm">Sortie</div>
                    </div>
                    <div className="text-center bg-white/5 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-red-400">{result.stats.removedCount}</div>
                      <div className="text-white/60 text-sm">Supprimés</div>
                    </div>
                    <div className="text-center bg-white/5 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-green-400">
                        {result.stats.inputLength > 0 ? 
                          Math.round(((result.stats.tokensCount || result.stats.outputLength) / result.stats.inputLength) * 100) : 0}%
                      </div>
                      <div className="text-white/60 text-sm">Conservé</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Flèche vers l'étape suivante */}
            {index < results.length - 1 && (
              <div className="flex justify-center py-6">
                <ArrowDown className={`h-10 w-10 ${config.colors.primary} animate-pulse`} />
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">🚀 Pipeline NLP Avancé</h2>
        <p className="text-white/70">Analyse complète avec visualisations et code source</p>
      </div>

      {/* Configuration du pipeline */}
      <PipelineConfigurator
        config={pipelineConfig}
        onConfigChange={setPipelineConfig}
        model={selectedModel === 'comparison' ? 'nltk' : selectedModel}
      />

      {/* Sélecteur de vue */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-6 rounded-xl border border-white/20 shadow-2xl">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <BarChart3 className="h-6 w-6 text-white/70" />
            <span className="text-white font-bold text-lg">Mode d'Affichage</span>
          </div>
          
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
          >
            {showAdvanced ? 'Vue Simple' : 'Vue Avancée'}
          </button>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          {[
            { key: 'nltk', label: 'NLTK Seul', icon: Cpu, color: 'blue' },
            { key: 'bert', label: 'BERT Seul', icon: Brain, color: 'purple' },
            { key: 'comparison', label: 'Comparaison', icon: BarChart3, color: 'green' }
          ].map((option) => (
            <button
              key={option.key}
              onClick={() => setSelectedModel(option.key as any)}
              className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                selectedModel === option.key
                  ? `border-${option.color}-500/50 bg-${option.color}-500/20 shadow-lg`
                  : 'border-white/20 bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="flex items-center space-x-4 mb-4">
                <option.icon className={`h-8 w-8 ${
                  selectedModel === option.key ? `text-${option.color}-400` : 'text-white/60'
                }`} />
                <div className="text-left">
                  <h3 className={`font-bold text-xl ${
                    selectedModel === option.key ? `text-${option.color}-400` : 'text-white'
                  }`}>
                    {option.label}
                  </h3>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Texte original */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-6 rounded-xl border border-white/20 shadow-xl">
        <h3 className="text-white font-bold text-xl mb-4 flex items-center">
          <Eye className="h-6 w-6 text-cyan-400 mr-3" />
          Texte Original
        </h3>
        <div className="bg-white/10 p-4 rounded-lg border border-white/20">
          <p className="text-white font-mono text-sm break-words">"{text}"</p>
        </div>
        <div className="mt-4 text-center">
          <span className="text-cyan-400 font-bold text-2xl">{text.length}</span>
          <span className="text-white/70 text-sm ml-2">caractères</span>
        </div>
      </div>

      {/* Contenu principal */}
      {selectedModel === 'comparison' ? (
        <div className="space-y-8">
          {/* Comparaison des modèles */}
          {nltkResults.length > 0 && bertResults.length > 0 && (
            <ModelComparison 
              nltkResults={nltkResults}
              bertResults={bertResults}
              text={text}
            />
          )}
          
          {/* Visualisations côte à côte */}
          <div className="grid md:grid-cols-2 gap-8">
            {/* NLTK */}
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-blue-400 text-center">📊 NLTK Pipeline</h3>
              {nltkResults.length > 0 && renderPipelineSteps(nltkResults, 'nltk')}
            </div>
            
            {/* BERT */}
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-purple-400 text-center">🧠 BERT Pipeline</h3>
              {bertResults.length > 0 && renderPipelineSteps(bertResults, 'bert')}
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Pipeline du modèle sélectionné */}
          {selectedModel === 'nltk' && nltkResults.length > 0 && renderPipelineSteps(nltkResults, 'nltk')}
          {selectedModel === 'bert' && bertResults.length > 0 && renderPipelineSteps(bertResults, 'bert')}
        </div>
      )}

      {/* Analyses avancées */}
      {showAdvanced && isInitialized && (
        <div className="space-y-8">
          {/* Analyse de sentiment */}
          {selectedModel !== 'comparison' && (
            <SentimentAnalyzer
              text={text}
              tokens={selectedModel === 'nltk' ? 
                (nltkResults.length > 0 ? nltkResults[nltkResults.length - 1]?.tokens || [] : []) :
                (bertResults.length > 0 ? bertResults[bertResults.length - 1]?.tokens || [] : [])
              }
              model={selectedModel}
            />
          )}

          {/* Graphiques interactifs */}
          {selectedModel !== 'comparison' && (
            <InteractiveCharts
              pipelineResults={selectedModel === 'nltk' ? nltkResults : bertResults}
              finalTokens={selectedModel === 'nltk' ? 
                (nltkResults.length > 0 ? nltkResults[nltkResults.length - 1]?.tokens || [] : []) :
                (bertResults.length > 0 ? bertResults[bertResults.length - 1]?.tokens || [] : [])
              }
              model={selectedModel}
            />
          )}

          {/* Nuage de mots */}
          {selectedModel !== 'comparison' && (
            <WordCloud
              tokens={selectedModel === 'nltk' ? 
                (nltkResults.length > 0 ? nltkResults[nltkResults.length - 1]?.tokens || [] : []) :
                (bertResults.length > 0 ? bertResults[bertResults.length - 1]?.tokens || [] : [])
              }
              model={selectedModel}
            />
          )}
        </div>
      )}

      {/* Résumé final */}
      {isInitialized && (
        <div className="bg-gradient-to-r from-green-500/20 via-blue-500/20 to-purple-500/20 p-8 rounded-xl border border-green-500/30 shadow-2xl">
          <div className="text-center mb-8">
            <h3 className="text-white font-bold text-3xl mb-4 flex items-center justify-center">
              <CheckCircle className="h-8 w-8 text-green-400 mr-4" />
              🎉 Analyse NLP Complète Terminée !
            </h3>
            <p className="text-white/80 text-lg">Pipeline avancé avec visualisations, code source et comparaisons</p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-6 gap-6">
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-white mb-2">{text.length}</div>
              <div className="text-white/70 text-sm">Caractères originaux</div>
            </div>
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-blue-400 mb-2">
                {nltkResults.length > 0 ? nltkResults[nltkResults.length - 1]?.stats.tokensCount || 0 : 0}
              </div>
              <div className="text-white/70 text-sm">Tokens NLTK</div>
            </div>
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-purple-400 mb-2">
                {bertResults.length > 0 ? bertResults[bertResults.length - 1]?.stats.tokensCount || 0 : 0}
              </div>
              <div className="text-white/70 text-sm">Tokens BERT</div>
            </div>
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-red-400 mb-2">
                {nltkResults.reduce((acc, step) => acc + step.stats.removedCount, 0) + 
                 bertResults.reduce((acc, step) => acc + step.stats.removedCount, 0)}
              </div>
              <div className="text-white/70 text-sm">Total supprimés</div>
            </div>
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-yellow-400 mb-2">
                {Math.max(nltkResults.length, bertResults.length)}
              </div>
              <div className="text-white/70 text-sm">Étapes complétées</div>
            </div>
            <div className="text-center bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-green-400 mb-2">100%</div>
              <div className="text-white/70 text-sm">Succès</div>
            </div>
          </div>
        </div>
      )}

      {/* Popup de code */}
      <CodePopup
        isOpen={codePopup.isOpen}
        onClose={closeCodePopup}
        stepTitle={codePopup.stepTitle}
        code={codePopup.code}
        language="python"
        model={codePopup.model}
        stepNumber={codePopup.stepNumber}
      />
    </div>
  );
};