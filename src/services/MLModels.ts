// Services pour les modèles de Machine Learning réels
export interface SentimentResult {
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
  emotions?: {
    joy: number;
    anger: number;
    fear: number;
    sadness: number;
    surprise: number;
    disgust: number;
  };
}

export interface TokenClassification {
  token: string;
  label: string;
  confidence: number;
  start: number;
  end: number;
}

// Classe pour gérer les modèles Hugging Face
export class HuggingFaceModels {
  private apiKey: string | null = null;
  private baseUrl = 'https://api-inference.huggingface.co/models';

  constructor(apiKey?: string) {
    this.apiKey = apiKey || null;
  }

  private async makeRequest(modelName: string, inputs: any, options: any = {}) {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.baseUrl}/${modelName}`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        inputs,
        options: {
          wait_for_model: true,
          use_cache: false,
          ...options
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Erreur API Hugging Face: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  // Analyse de sentiment avec BERT - VERSION CORRIGÉE
  async analyzeSentimentBERT(text: string): Promise<SentimentResult> {
    try {
      // Essayer plusieurs modèles BERT pour maximiser les chances de succès
      const models = [
        'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'nlptown/bert-base-multilingual-uncased-sentiment',
        'distilbert-base-uncased-finetuned-sst-2-english'
      ];

      let result = null;
      let lastError = null;

      for (const model of models) {
        try {
          console.log(`Tentative avec le modèle: ${model}`);
          result = await this.makeRequest(model, text);
          console.log(`Succès avec ${model}:`, result);
          break;
        } catch (error) {
          console.warn(`Échec avec ${model}:`, error);
          lastError = error;
          continue;
        }
      }

      if (!result) {
        throw lastError || new Error('Tous les modèles BERT ont échoué');
      }

      // Traitement des résultats selon le format du modèle
      if (Array.isArray(result) && result.length > 0) {
        const scores = result[0];
        
        // Format pour twitter-roberta-base-sentiment-latest
        if (scores.some((s: any) => s.label && s.label.startsWith('LABEL_'))) {
          const labelMap: Record<string, 'positive' | 'negative' | 'neutral'> = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
          };

          const normalizedScores = {
            negative: scores.find((s: any) => s.label === 'LABEL_0')?.score || 0,
            neutral: scores.find((s: any) => s.label === 'LABEL_1')?.score || 0,
            positive: scores.find((s: any) => s.label === 'LABEL_2')?.score || 0
          };

          const maxScore = Math.max(...Object.values(normalizedScores));
          const prediction = Object.entries(normalizedScores).find(([_, score]) => score === maxScore);
          
          return {
            label: prediction?.[0] as 'positive' | 'negative' | 'neutral' || 'neutral',
            confidence: maxScore,
            scores: normalizedScores
          };
        }
        
        // Format pour distilbert-base-uncased-finetuned-sst-2-english
        else if (scores.some((s: any) => ['POSITIVE', 'NEGATIVE'].includes(s.label))) {
          const positive = scores.find((s: any) => s.label === 'POSITIVE')?.score || 0;
          const negative = scores.find((s: any) => s.label === 'NEGATIVE')?.score || 0;
          const neutral = Math.max(0, 1 - positive - negative);

          const maxScore = Math.max(positive, negative, neutral);
          let label: 'positive' | 'negative' | 'neutral' = 'neutral';
          
          if (maxScore === positive) label = 'positive';
          else if (maxScore === negative) label = 'negative';

          return {
            label,
            confidence: maxScore,
            scores: { positive, negative, neutral }
          };
        }
        
        // Format pour nlptown/bert-base-multilingual-uncased-sentiment (1-5 étoiles)
        else if (scores.some((s: any) => s.label && /^[1-5] star/.test(s.label))) {
          const starScores = scores.reduce((acc: any, s: any) => {
            const stars = parseInt(s.label.charAt(0));
            acc[stars] = s.score;
            return acc;
          }, {});

          // Convertir les étoiles en sentiment
          const positive = (starScores[4] || 0) + (starScores[5] || 0);
          const negative = (starScores[1] || 0) + (starScores[2] || 0);
          const neutral = starScores[3] || 0;

          const maxScore = Math.max(positive, negative, neutral);
          let label: 'positive' | 'negative' | 'neutral' = 'neutral';
          
          if (maxScore === positive) label = 'positive';
          else if (maxScore === negative) label = 'negative';

          return {
            label,
            confidence: maxScore,
            scores: { positive, negative, neutral }
          };
        }
      }

      // Si aucun format reconnu, utiliser le fallback
      throw new Error('Format de réponse non reconnu');

    } catch (error) {
      console.warn('Erreur modèle BERT, utilisation du fallback:', error);
      return this.fallbackSentimentAnalysis(text);
    }
  }

  // Analyse de sentiment avec DistilBERT - VERSION AMÉLIORÉE
  async analyzeSentimentDistilBERT(text: string): Promise<SentimentResult> {
    try {
      const result = await this.makeRequest(
        'distilbert-base-uncased-finetuned-sst-2-english',
        text
      );

      if (Array.isArray(result) && result.length > 0) {
        const scores = result[0];
        const positive = scores.find((s: any) => s.label === 'POSITIVE')?.score || 0;
        const negative = scores.find((s: any) => s.label === 'NEGATIVE')?.score || 0;
        const neutral = Math.max(0, 1 - positive - negative);

        const maxScore = Math.max(positive, negative, neutral);
        let label: 'positive' | 'negative' | 'neutral' = 'neutral';
        
        if (maxScore === positive) label = 'positive';
        else if (maxScore === negative) label = 'negative';

        return {
          label,
          confidence: maxScore,
          scores: { positive, negative, neutral }
        };
      }

      throw new Error('Format de réponse invalide');
    } catch (error) {
      console.warn('Erreur modèle DistilBERT, utilisation du fallback:', error);
      return this.fallbackSentimentAnalysis(text);
    }
  }

  // Analyse d'émotions avec modèle spécialisé
  async analyzeEmotions(text: string): Promise<SentimentResult> {
    try {
      const result = await this.makeRequest(
        'j-hartmann/emotion-english-distilroberta-base',
        text
      );

      if (Array.isArray(result) && result.length > 0) {
        const emotions = result[0];
        const emotionScores = {
          joy: emotions.find((e: any) => e.label === 'joy')?.score || 0,
          anger: emotions.find((e: any) => e.label === 'anger')?.score || 0,
          fear: emotions.find((e: any) => e.label === 'fear')?.score || 0,
          sadness: emotions.find((e: any) => e.label === 'sadness')?.score || 0,
          surprise: emotions.find((e: any) => e.label === 'surprise')?.score || 0,
          disgust: emotions.find((e: any) => e.label === 'disgust')?.score || 0
        };

        // Convertir les émotions en sentiment global
        const positiveEmotions = emotionScores.joy + emotionScores.surprise;
        const negativeEmotions = emotionScores.anger + emotionScores.fear + emotionScores.sadness + emotionScores.disgust;
        const neutral = Math.max(0, 1 - positiveEmotions - negativeEmotions);

        let label: 'positive' | 'negative' | 'neutral' = 'neutral';
        let confidence = 0;

        if (positiveEmotions > negativeEmotions && positiveEmotions > neutral) {
          label = 'positive';
          confidence = positiveEmotions;
        } else if (negativeEmotions > positiveEmotions && negativeEmotions > neutral) {
          label = 'negative';
          confidence = negativeEmotions;
        } else {
          confidence = neutral;
        }

        return {
          label,
          confidence,
          scores: {
            positive: positiveEmotions,
            negative: negativeEmotions,
            neutral
          },
          emotions: emotionScores
        };
      }

      throw new Error('Format de réponse invalide');
    } catch (error) {
      console.warn('Erreur modèle émotions, utilisation du fallback:', error);
      return this.fallbackSentimentAnalysis(text);
    }
  }

  // Classification de tokens avec NER
  async classifyTokens(text: string): Promise<TokenClassification[]> {
    try {
      const result = await this.makeRequest(
        'dbmdz/bert-large-cased-finetuned-conll03-english',
        text
      );

      if (Array.isArray(result)) {
        return result.map((token: any) => ({
          token: token.word,
          label: token.entity,
          confidence: token.score,
          start: token.start,
          end: token.end
        }));
      }

      return [];
    } catch (error) {
      console.warn('Erreur classification tokens:', error);
      return [];
    }
  }

  // Fallback amélioré basé sur des patterns linguistiques
  private fallbackSentimentAnalysis(text: string): SentimentResult {
    const words = text.toLowerCase().split(/\s+/);
    
    // Dictionnaire de sentiments plus complet
    const sentimentWords = {
      // Très positifs
      'amazing': 0.9, 'incredible': 0.85, 'outstanding': 0.9, 'excellent': 0.8, 'fantastic': 0.85,
      'perfect': 0.9, 'wonderful': 0.8, 'brilliant': 0.8, 'superb': 0.75, 'magnificent': 0.85,
      'awesome': 0.7, 'great': 0.6, 'good': 0.5, 'nice': 0.4, 'fine': 0.3,
      'love': 0.75, 'adore': 0.8, 'recommend': 0.6, 'satisfied': 0.55, 'happy': 0.65,
      
      // Très négatifs
      'terrible': -0.9, 'horrible': -0.85, 'awful': -0.8, 'disgusting': -0.9, 'pathetic': -0.8,
      'worst': -0.9, 'hate': -0.8, 'despise': -0.85, 'useless': -0.75, 'worthless': -0.8,
      'bad': -0.6, 'poor': -0.5, 'disappointing': -0.65, 'broken': -0.7, 'frustrated': -0.6,
      
      // Modérés
      'okay': 0.1, 'decent': 0.3, 'average': 0.0, 'normal': 0.0, 'standard': 0.1
    };

    let totalScore = 0;
    let wordCount = 0;

    // Analyse avec gestion des négations et intensificateurs
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      let score = sentimentWords[word] || 0;
      
      if (Math.abs(score) > 0) {
        // Vérifier les négations
        if (i > 0 && ['not', 'never', 'no', 'nothing'].includes(words[i-1])) {
          score *= -1;
        }
        
        // Vérifier les intensificateurs
        if (i > 0 && ['very', 'extremely', 'incredibly', 'absolutely'].includes(words[i-1])) {
          score *= 1.5;
        }
        
        totalScore += score;
        wordCount++;
      }
    }

    const avgScore = wordCount > 0 ? totalScore / wordCount : 0;
    
    // Calculer les scores normalisés
    const positive = Math.max(0, avgScore);
    const negative = Math.max(0, -avgScore);
    const neutral = Math.max(0, 1 - Math.abs(avgScore));

    let label: 'positive' | 'negative' | 'neutral' = 'neutral';
    let confidence = 0.5;

    if (avgScore > 0.1) {
      label = 'positive';
      confidence = Math.min(0.9, positive + 0.1);
    } else if (avgScore < -0.1) {
      label = 'negative';
      confidence = Math.min(0.9, negative + 0.1);
    } else {
      confidence = Math.min(0.9, neutral + 0.1);
    }

    return {
      label,
      confidence,
      scores: { positive, negative, neutral }
    };
  }
}

// Classe pour les modèles locaux (simulation d'un modèle NLTK entraîné)
export class LocalNLTKModel {
  private isLoaded = false;
  private modelWeights: Map<string, number> = new Map();

  async loadModel(): Promise<void> {
    if (this.isLoaded) return;

    // Simulation du chargement d'un modèle pré-entraîné
    await new Promise(resolve => setTimeout(resolve, 500));
    
    this.initializeModelWeights();
    this.isLoaded = true;
  }

  private initializeModelWeights(): void {
    // Simulation de poids appris par un modèle de régression logistique
    const learnedWeights = new Map([
      ['feature_positive_words', 2.3],
      ['feature_exclamation', 1.8],
      ['feature_superlatives', 2.1],
      ['feature_recommendation', 1.9],
      ['feature_negative_words', -2.5],
      ['feature_complaints', -2.2],
      ['feature_problems', -1.9],
      ['feature_regret', -2.0],
      ['feature_negation', -1.5],
      ['feature_intensifiers', 1.3],
      ['feature_questions', -0.5],
      ['feature_length', 0.1]
    ]);

    this.modelWeights = learnedWeights;
  }

  async predict(text: string): Promise<SentimentResult> {
    await this.loadModel();

    const features = this.extractFeatures(text);
    
    let score = 0;
    for (const [feature, value] of features.entries()) {
      const weight = this.modelWeights.get(feature) || 0;
      score += weight * value;
    }

    // Application de la fonction sigmoïde
    const probability = 1 / (1 + Math.exp(-score));
    
    let label: 'positive' | 'negative' | 'neutral' = 'neutral';
    let confidence = 0.5;

    if (probability > 0.6) {
      label = 'positive';
      confidence = probability;
    } else if (probability < 0.4) {
      label = 'negative';
      confidence = 1 - probability;
    } else {
      confidence = 1 - Math.abs(probability - 0.5) * 2;
    }

    return {
      label,
      confidence,
      scores: {
        positive: probability,
        negative: 1 - probability,
        neutral: 1 - Math.abs(probability - 0.5) * 2
      }
    };
  }

  private extractFeatures(text: string): Map<string, number> {
    const features = new Map<string, number>();
    const words = text.toLowerCase().split(/\s+/);
    const textLength = text.length;

    // Features basées sur des patterns réels
    const positiveWords = words.filter(word => 
      /\b(good|great|excellent|amazing|wonderful|fantastic|perfect|love|awesome|brilliant|outstanding|superb|magnificent|incredible|exceptional|marvelous)\b/.test(word)
    ).length;
    features.set('feature_positive_words', positiveWords / words.length);

    const negativeWords = words.filter(word => 
      /\b(bad|terrible|awful|horrible|worst|hate|poor|disappointing|useless|worthless|pathetic|disgusting|appalling|dreadful|atrocious)\b/.test(word)
    ).length;
    features.set('feature_negative_words', negativeWords / words.length);

    const exclamations = (text.match(/!/g) || []).length;
    features.set('feature_exclamation', exclamations / textLength * 100);

    const superlatives = words.filter(word => 
      /\b(best|worst|most|least|greatest|smallest|largest|finest|perfect|ultimate|supreme)\b/.test(word)
    ).length;
    features.set('feature_superlatives', superlatives / words.length);

    const recommendations = words.filter(word => 
      /\b(recommend|suggest|advise|endorse|propose|urge)\b/.test(word)
    ).length;
    features.set('feature_recommendation', recommendations / words.length);

    const complaints = words.filter(word => 
      /\b(complain|complaint|problem|issue|trouble|difficulty|concern|worry)\b/.test(word)
    ).length;
    features.set('feature_complaints', complaints / words.length);

    const regret = words.filter(word => 
      /\b(regret|mistake|wrong|error|sorry|apologize|disappointed|frustrated)\b/.test(word)
    ).length;
    features.set('feature_regret', regret / words.length);

    const negations = words.filter(word => 
      /\b(not|never|no|nothing|none|neither|nor|hardly|barely|scarcely)\b/.test(word)
    ).length;
    features.set('feature_negation', negations / words.length);

    const intensifiers = words.filter(word => 
      /\b(very|extremely|incredibly|absolutely|totally|completely|utterly|thoroughly|highly|deeply|really|quite|pretty|rather|fairly|super|ultra|mega)\b/.test(word)
    ).length;
    features.set('feature_intensifiers', intensifiers / words.length);

    const questions = (text.match(/\?/g) || []).length;
    features.set('feature_questions', questions / textLength * 100);

    features.set('feature_length', Math.min(1, textLength / 500));

    return features;
  }
}

// Factory pour créer les modèles appropriés
export class ModelFactory {
  static createHuggingFaceModel(apiKey?: string): HuggingFaceModels {
    return new HuggingFaceModels(apiKey);
  }

  static createLocalNLTKModel(): LocalNLTKModel {
    return new LocalNLTKModel();
  }

  // Méthode pour tester la disponibilité des modèles
  static async testModelAvailability(): Promise<{
    huggingFace: boolean;
    local: boolean;
  }> {
    const results = {
      huggingFace: false,
      local: true
    };

    try {
      const hfModel = new HuggingFaceModels();
      // Test avec un texte simple
      const testResult = await hfModel.analyzeSentimentBERT("This is a test");
      if (testResult && testResult.label) {
        results.huggingFace = true;
        console.log('Hugging Face disponible:', testResult);
      }
    } catch (error) {
      console.log('Hugging Face non disponible, utilisation du modèle local:', error);
    }

    return results;
  }
}