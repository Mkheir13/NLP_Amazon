import { BERTTrainingService } from './BERTTrainingService';

export interface RealNLPAnalysis {
  text: string;
  nltk: {
    sentiment: 'positive' | 'negative' | 'neutral';
    confidence: number;
    scores: {
      neg: number;
      neu: number;
      pos: number;
      compound: number;
    };
    polarity: number;
  };
  bert?: {
    sentiment: 'positive' | 'negative';
    confidence: number;
    class: number;
  };
  features: {
    wordCount: number;
    charCount: number;
    sentenceCount: number;
    positiveWords: number;
    negativeWords: number;
    emotionalWords: number;
  };
  keywords: { [key: string]: number };
  comparison?: {
    agreement: boolean;
    nltkConfidence: number;
    bertConfidence: number;
    finalSentiment: 'positive' | 'negative' | 'neutral';
    reasoning: string;
  };
}

export class RealNLPService {
  
  // Analyse complète avec NLTK + BERT optionnel
  static async analyzeWithRealNLP(
    text: string, 
    bertModelId?: string
  ): Promise<RealNLPAnalysis> {
    try {
      // 1. Analyse NLTK (toujours)
      const nltkResult = await BERTTrainingService.analyzeWithNLTK(text);
      
      // 2. Analyse BERT (si modèle disponible)
      let bertResult = undefined;
      if (bertModelId) {
        try {
          bertResult = await BERTTrainingService.predictWithBERT(bertModelId, text);
        } catch (error) {
          console.warn('BERT non disponible, utilisation NLTK seulement');
        }
      }
      
      // 3. Extraction des features basiques
      const features = this.extractFeatures(text);
      
      // 4. Extraction des mots-clés
      const keywords = this.extractKeywords(text);
      
      // 5. Comparaison NLTK vs BERT (si BERT disponible)
      const comparison = bertResult ? this.compareResults(nltkResult, bertResult) : undefined;
      
      return {
        text,
        nltk: nltkResult,
        bert: bertResult,
        features,
        keywords,
        comparison
      };
      
    } catch (error) {
      console.error('Erreur analyse NLP:', error);
      throw new Error(`Erreur lors de l'analyse: ${error instanceof Error ? error.message : 'Erreur inconnue'}`);
    }
  }
  
  // Extraction de features linguistiques
  private static extractFeatures(text: string) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Dictionnaires pour compter les mots émotionnels
    const positiveWords = [
      'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
      'perfect', 'love', 'awesome', 'brilliant', 'outstanding', 'superb'
    ];
    
    const negativeWords = [
      'bad', 'terrible', 'horrible', 'awful', 'disgusting', 'pathetic',
      'worst', 'hate', 'useless', 'disappointing', 'broken', 'defective'
    ];
    
    const emotionalWords = [...positiveWords, ...negativeWords];
    
    const lowerText = text.toLowerCase();
    
    return {
      wordCount: words.length,
      charCount: text.length,
      sentenceCount: sentences.length,
      positiveWords: positiveWords.filter(word => lowerText.includes(word)).length,
      negativeWords: negativeWords.filter(word => lowerText.includes(word)).length,
      emotionalWords: emotionalWords.filter(word => lowerText.includes(word)).length
    };
  }
  
  // Extraction de mots-clés
  private static extractKeywords(text: string): { [key: string]: number } {
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 3);
      
    const stopWords = new Set([
      'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
      'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
      'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
      'such', 'take', 'than', 'them', 'well', 'were'
    ]);
    
    const keywords: { [key: string]: number } = {};
    
    words.forEach(word => {
      if (!stopWords.has(word)) {
        keywords[word] = (keywords[word] || 0) + 1;
      }
    });
    
    // Retourner les 10 mots les plus fréquents
    return Object.fromEntries(
      Object.entries(keywords)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10)
    );
  }
  
  // Comparaison NLTK vs BERT
  private static compareResults(
    nltk: any, 
    bert: any
  ) {
    const nltkSentiment = nltk.sentiment;
    const bertSentiment = bert.sentiment;
    const agreement = nltkSentiment === bertSentiment;
    
    const nltkConfidence = nltk.confidence;
    const bertConfidence = bert.confidence;
    
    // Logique de décision finale
    let finalSentiment: 'positive' | 'negative' | 'neutral';
    let reasoning: string;
    
    if (agreement) {
      finalSentiment = nltkSentiment;
      reasoning = `NLTK et BERT sont d'accord: ${nltkSentiment}`;
    } else {
      // En cas de désaccord, privilégier le plus confiant
      if (bertConfidence > nltkConfidence) {
        finalSentiment = bertSentiment;
        reasoning = `Désaccord: BERT plus confiant (${(bertConfidence * 100).toFixed(1)}% vs ${(nltkConfidence * 100).toFixed(1)}%)`;
      } else {
        finalSentiment = nltkSentiment;
        reasoning = `Désaccord: NLTK plus confiant (${(nltkConfidence * 100).toFixed(1)}% vs ${(bertConfidence * 100).toFixed(1)}%)`;
      }
    }
    
    return {
      agreement,
      nltkConfidence,
      bertConfidence,
      finalSentiment,
      reasoning
    };
  }
  
  // Vérifier si le backend est disponible
  static async isBackendAvailable(): Promise<boolean> {
    return await BERTTrainingService.checkBackendHealth();
  }
  
  // Obtenir les modèles BERT disponibles
  static async getAvailableBERTModels() {
    try {
      return await BERTTrainingService.getBERTModels();
    } catch (error) {
      return [];
    }
  }
} 