import { Review } from './DatasetLoader';
import ConfigManager from '../config/AppConfig';

export interface BERTTrainingConfig {
  model_name: 'distilbert-base-uncased' | 'bert-base-uncased' | 'roberta-base';
  epochs: number;
  batch_size: number;
  learning_rate: number;
  test_size: number;
}

export interface BERTModel {
  id: string;
  name: string;
  type: 'bert';
  model_name: string;
  config: BERTTrainingConfig;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    eval_loss: number;
  };
  trained_on: number;
  created_at: string;
}

export interface NLTKResult {
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  scores: {
    neg: number;
    neu: number;
    pos: number;
    compound: number;
  };
  polarity: number;
}

export class BERTTrainingService {
  private static readonly API_BASE = ConfigManager.getApiUrl('bert').replace('/api/bert', '/api');

  // Vérifier si le backend est disponible
  static async checkBackendHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.API_BASE}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch (error) {
      console.error('Backend non disponible:', error);
      return false;
    }
  }

  // Entraîner un modèle BERT
  static async trainBERTModel(
    reviews: Review[],
    config: BERTTrainingConfig,
    onProgress?: (message: string) => void
  ): Promise<BERTModel> {
    try {
      onProgress?.('Vérification du backend...');
      
      const isHealthy = await this.checkBackendHealth();
      if (!isHealthy) {
        throw new Error('Backend Python non disponible. Assurez-vous que le serveur Flask est démarré.');
      }

      onProgress?.('Envoi des données au backend...');

      // Préparer les données
      const trainingData = reviews.map(review => ({
        text: review.text,
        label: review.label,
        sentiment: review.sentiment
      }));

      const response = await fetch(`${this.API_BASE}/train/bert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: trainingData,
          config: config
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de l\'entraînement');
      }

      const result = await response.json();
      onProgress?.('Modèle BERT entraîné avec succès !');
      
      return result.model;
    } catch (error) {
      console.error('Erreur entraînement BERT:', error);
      throw error;
    }
  }

  // Analyser avec NLTK
  static async analyzeWithNLTK(text: string): Promise<NLTKResult> {
    try {
      const response = await fetch(`${this.API_BASE}/analyze/nltk`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de l\'analyse NLTK');
      }

      const result = await response.json();
      return result.result;
    } catch (error) {
      console.error('Erreur analyse NLTK:', error);
      throw error;
    }
  }

  // Analyser plusieurs textes avec NLTK
  static async batchAnalyzeWithNLTK(texts: string[]): Promise<NLTKResult[]> {
    try {
      const response = await fetch(`${this.API_BASE}/analyze/nltk/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de l\'analyse NLTK en lot');
      }

      const result = await response.json();
      return result.results;
    } catch (error) {
      console.error('Erreur analyse NLTK batch:', error);
      throw error;
    }
  }

  // Récupérer la liste des modèles BERT entraînés
  static async getBERTModels(): Promise<BERTModel[]> {
    try {
      const response = await fetch(`${this.API_BASE}/models`);
      
      if (!response.ok) {
        throw new Error('Erreur lors de la récupération des modèles');
      }

      const result = await response.json();
      return result.models;
    } catch (error) {
      console.error('Erreur récupération modèles:', error);
      return [];
    }
  }

  // Prédire avec un modèle BERT entraîné
  static async predictWithBERT(modelId: string, text: string): Promise<{
    sentiment: 'positive' | 'negative';
    confidence: number;
    class: number;
  }> {
    try {
      const response = await fetch(`${this.API_BASE}/predict/bert/${modelId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de la prédiction');
      }

      const result = await response.json();
      return result.prediction;
    } catch (error) {
      console.error('Erreur prédiction BERT:', error);
      throw error;
    }
  }

  // Comparer les performances NLTK vs BERT
  static async compareAnalysis(text: string, bertModelId?: string): Promise<{
    nltk: NLTKResult;
    bert?: {
      sentiment: 'positive' | 'negative';
      confidence: number;
      class: number;
    };
  }> {
    try {
      const promises: Promise<any>[] = [this.analyzeWithNLTK(text)];
      
      if (bertModelId) {
        promises.push(this.predictWithBERT(bertModelId, text));
      }

      const results = await Promise.all(promises);
      
      return {
        nltk: results[0],
        bert: results[1] || undefined
      };
    } catch (error) {
      console.error('Erreur comparaison:', error);
      throw error;
    }
  }
} 