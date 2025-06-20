export interface WordEmbedding {
  word: string;
  embedding: number[];
  dimension: number;
}

export interface SentenceEmbedding {
  text: string;
  embedding: number[];
  dimension: number;
}

export interface SimilarWord {
  word: string;
  similarity: number;
}

export interface SemanticSearchResult {
  index: number;
  text: string;
  similarity: number;
  text_preview: string;
}

export interface EmbeddingVisualization {
  plot: string; // JSON string of Plotly figure
  method: string;
  words_count: number;
  words_found: string[];
  words_not_found: string[];
}

export interface EmbeddingModel {
  id: string;
  type: string;
  path: string;
  config: any;
  vocabulary_size: number;
  trained_on: number;
  created_at: string;
}

export interface EmbeddingStats {
  vocabulary_size: number;
  vector_size: number;
  most_frequent_words: [string, number][];
  model_type: string;
}

export interface SemanticAnalysis {
  text: string;
  sentence_embedding_shape: number[];
  sentence_embedding_norm: number;
  word_count: number;
  unique_words: number;
  words_in_vocabulary: number;
  word_similarities: { [key: string]: [string, number][] };
  semantic_density: number;
}

export class EmbeddingService {
  private static readonly BASE_URL = 'http://localhost:5000/api/embeddings';

  // Entraîner un modèle TF-IDF
  static async trainTFIDF(texts: string[]): Promise<{ stats: any }> {
    try {
      const response = await fetch(`${this.BASE_URL}/train/tfidf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de l\'entraînement TF-IDF');
      }

      return { stats: data.stats };
    } catch (error) {
      console.error('Erreur entraînement TF-IDF:', error);
      throw error;
    }
  }

  // Entraîner un modèle Word2Vec
  static async trainWord2Vec(texts: string[], config: any = {}): Promise<EmbeddingModel> {
    try {
      const response = await fetch(`${this.BASE_URL}/train/word2vec`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts, config }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de l\'entraînement');
      }

      return data.model;
    } catch (error) {
      console.error('Erreur entraînement Word2Vec:', error);
      throw error;
    }
  }

  // Obtenir l'embedding d'un mot
  static async getWordEmbedding(word: string, modelId?: string): Promise<WordEmbedding> {
    try {
      const params = new URLSearchParams();
      if (modelId) params.append('model_id', modelId);

      const response = await fetch(`${this.BASE_URL}/word/${encodeURIComponent(word)}?${params}`);

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`Mot "${word}" non trouvé dans le vocabulaire`);
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la récupération de l\'embedding');
      }

      return {
        word: data.word,
        embedding: data.embedding,
        dimension: data.dimension
      };
    } catch (error) {
      console.error('Erreur récupération embedding mot:', error);
      throw error;
    }
  }

  // Obtenir l'embedding d'une phrase
  static async getSentenceEmbedding(text: string): Promise<SentenceEmbedding> {
    try {
      const response = await fetch(`${this.BASE_URL}/sentence`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la récupération de l\'embedding');
      }

      return {
        text: data.text,
        embedding: data.embedding,
        dimension: data.dimension
      };
    } catch (error) {
      console.error('Erreur récupération embedding phrase:', error);
      throw error;
    }
  }

  // Trouver des mots similaires
  static async findSimilarWords(word: string, topK: number = 10, modelId?: string): Promise<SimilarWord[]> {
    try {
      const params = new URLSearchParams();
      params.append('top_k', topK.toString());
      if (modelId) params.append('model_id', modelId);

      const response = await fetch(`${this.BASE_URL}/similar/${encodeURIComponent(word)}?${params}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la recherche de similarité');
      }

      return data.similar_words.map(([word, similarity]: [string, number]) => ({
        word,
        similarity
      }));
    } catch (error) {
      console.error('Erreur recherche mots similaires:', error);
      throw error;
    }
  }

  // Recherche sémantique
  static async semanticSearch(query: string, texts: string[], topK: number = 5): Promise<SemanticSearchResult[]> {
    try {
      const response = await fetch(`${this.BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, texts, top_k: topK }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la recherche sémantique');
      }

      return data.results;
    } catch (error) {
      console.error('Erreur recherche sémantique:', error);
      throw error;
    }
  }

  // Visualiser les embeddings
  static async visualizeEmbeddings(
    texts: string[],
    labels?: string[],
    method: string = 'pca'
  ): Promise<EmbeddingVisualization> {
    const response = await fetch(`${this.BASE_URL}/visualize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        texts,
        labels,
        method
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.visualization;
  }

  // Analyser la sémantique d'un texte
  static async analyzeTextSemantics(text: string): Promise<SemanticAnalysis> {
    try {
      const response = await fetch(`${this.BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de l\'analyse sémantique');
      }

      return data.analysis;
    } catch (error) {
      console.error('Erreur analyse sémantique:', error);
      throw error;
    }
  }

  // Obtenir la liste des modèles
  static async getEmbeddingModels(): Promise<EmbeddingModel[]> {
    try {
      const response = await fetch(`${this.BASE_URL}/models`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la récupération des modèles');
      }

      return data.models;
    } catch (error) {
      console.error('Erreur récupération modèles:', error);
      throw error;
    }
  }

  // Obtenir les statistiques d'un modèle
  static async getEmbeddingStats(modelId?: string): Promise<EmbeddingStats> {
    try {
      const params = new URLSearchParams();
      if (modelId) params.append('model_id', modelId);

      const response = await fetch(`${this.BASE_URL}/stats?${params}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de la récupération des statistiques');
      }

      return data.statistics;
    } catch (error) {
      console.error('Erreur récupération statistiques:', error);
      throw error;
    }
  }

  // Vérifier si le service d'embedding est disponible
  static async isEmbeddingServiceAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.BASE_URL}/models`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
} 