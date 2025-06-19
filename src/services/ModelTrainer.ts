import { Review } from './DatasetLoader';

export interface TrainingConfig {
  modelType: 'naive_bayes' | 'logistic_regression' | 'svm' | 'neural_network';
  testSize: number; // Pourcentage pour le test (0.1 = 10%)
  maxFeatures: number; // Nombre max de features pour la vectorisation
  ngrams: [number, number]; // N-grams (ex: [1, 2] pour unigrams et bigrams)
  epochs?: number; // Pour les réseaux de neurones
  learningRate?: number; // Pour les réseaux de neurones
}

export interface TrainingMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  trainingTime: number;
  predictions: Array<{
    text: string;
    predicted: number;
    actual: number;
    confidence: number;
  }>;
}

export interface TrainedModel {
  id: string;
  name: string;
  type: string;
  config: TrainingConfig;
  metrics: TrainingMetrics;
  vocabulary: string[];
  weights: number[] | number[][];
  createdAt: string;
  trainedOn: number; // Nombre d'exemples d'entraînement
}

export class ModelTrainer {
  private static models: TrainedModel[] = [];

  // Préprocessing du texte
  static preprocessText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Supprimer la ponctuation
      .replace(/\s+/g, ' ') // Normaliser les espaces
      .trim();
  }

  // Extraction de features TF-IDF simplifiée
  static extractFeatures(texts: string[], maxFeatures: number, ngrams: [number, number]): {
    features: number[][];
    vocabulary: string[];
  } {
    const vocabulary = new Set<string>();
    const termFreq = new Map<string, Map<string, number>>();

    // Extraire tous les n-grams
    texts.forEach((text, docIndex) => {
      const words = this.preprocessText(text).split(' ').filter(w => w.length > 2);
      const docTerms = new Map<string, number>();
      
      // Générer n-grams
      for (let n = ngrams[0]; n <= ngrams[1]; n++) {
        for (let i = 0; i <= words.length - n; i++) {
          const ngram = words.slice(i, i + n).join(' ');
          vocabulary.add(ngram);
          docTerms.set(ngram, (docTerms.get(ngram) || 0) + 1);
        }
      }
      
      termFreq.set(docIndex.toString(), docTerms);
    });

    // Limiter le vocabulaire aux termes les plus fréquents
    const termCounts = new Map<string, number>();
    vocabulary.forEach(term => {
      let count = 0;
      termFreq.forEach(doc => {
        if (doc.has(term)) count++;
      });
      termCounts.set(term, count);
    });

    const sortedTerms = Array.from(termCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxFeatures)
      .map(([term]) => term);

    // Calculer TF-IDF
    const features: number[][] = [];
    const docCount = texts.length;

    texts.forEach((text, docIndex) => {
      const docTerms = termFreq.get(docIndex.toString()) || new Map();
      const docFeatures: number[] = [];

      sortedTerms.forEach(term => {
        const tf = docTerms.get(term) || 0;
        const df = termCounts.get(term) || 1;
        const idf = Math.log(docCount / df);
        const tfidf = tf * idf;
        docFeatures.push(tfidf);
      });

      features.push(docFeatures);
    });

    return { features, vocabulary: sortedTerms };
  }

  // Naive Bayes
  static trainNaiveBayes(trainFeatures: number[][], trainLabels: number[]): {
    classPriors: number[];
    featureMeans: number[][];
    featureVars: number[][];
  } {
    const classes = [0, 1];
    const classPriors: number[] = [];
    const featureMeans: number[][] = [];
    const featureVars: number[][] = [];

    classes.forEach(cls => {
      const classIndices = trainLabels.map((label, idx) => label === cls ? idx : -1).filter(idx => idx !== -1);
      const classFeatures = classIndices.map(idx => trainFeatures[idx]);
      
      classPriors.push(classFeatures.length / trainFeatures.length);
      
      const means: number[] = [];
      const vars: number[] = [];
      
      for (let feature = 0; feature < trainFeatures[0].length; feature++) {
        const values = classFeatures.map(sample => sample[feature]);
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        
        means.push(mean);
        vars.push(Math.max(variance, 1e-9)); // Éviter division par zéro
      }
      
      featureMeans.push(means);
      featureVars.push(vars);
    });

    return { classPriors, featureMeans, featureVars };
  }

  // Régression logistique simplifiée
  static trainLogisticRegression(trainFeatures: number[][], trainLabels: number[], epochs = 100, learningRate = 0.01): {
    weights: number[];
    bias: number;
  } {
    const numFeatures = trainFeatures[0].length;
    let weights = new Array(numFeatures).fill(0).map(() => Math.random() * 0.01);
    let bias = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < trainFeatures.length; i++) {
        const features = trainFeatures[i];
        const label = trainLabels[i];
        
        // Forward pass
        const z = features.reduce((sum, feature, idx) => sum + feature * weights[idx], bias);
        const prediction = 1 / (1 + Math.exp(-z)); // Sigmoid
        
        // Backward pass
        const error = prediction - label;
        
        // Update weights
        for (let j = 0; j < weights.length; j++) {
          weights[j] -= learningRate * error * features[j];
        }
        bias -= learningRate * error;
      }
    }

    return { weights, bias };
  }

  // Prédiction
  static predict(model: TrainedModel, text: string): { prediction: number; confidence: number } {
    const processedText = this.preprocessText(text);
    const words = processedText.split(' ').filter(w => w.length > 2);
    
    // Créer le vecteur de features
    const features: number[] = new Array(model.vocabulary.length).fill(0);
    
    model.vocabulary.forEach((term, idx) => {
      const termWords = term.split(' ');
      let count = 0;
      
      for (let i = 0; i <= words.length - termWords.length; i++) {
        if (words.slice(i, i + termWords.length).join(' ') === term) {
          count++;
        }
      }
      
      features[idx] = count;
    });

    let prediction: number;
    let confidence: number;

    if (model.type === 'naive_bayes') {
      // Implémentation Naive Bayes
      const weights = model.weights as number[][];
      const priors = weights[0]; // classPriors
      const means = weights[1]; // featureMeans
      const vars = weights[2]; // featureVars
      
      let maxProb = -Infinity;
      prediction = 0;
      
      [0, 1].forEach(cls => {
        let logProb = Math.log(priors[cls]);
        
        features.forEach((feature, idx) => {
          const mean = means[cls * features.length + idx] || 0;
          const variance = vars[cls * features.length + idx] || 1e-9;
          const prob = Math.exp(-0.5 * Math.pow(feature - mean, 2) / variance) / Math.sqrt(2 * Math.PI * variance);
          logProb += Math.log(Math.max(prob, 1e-10));
        });
        
        if (logProb > maxProb) {
          maxProb = logProb;
          prediction = cls;
        }
      });
      
      confidence = Math.exp(maxProb) / (Math.exp(maxProb) + Math.exp(-maxProb));
    } else if (model.type === 'logistic_regression') {
      // Implémentation Régression Logistique
      const weights = model.weights as number[];
      const bias = weights[weights.length - 1];
      const w = weights.slice(0, -1);
      
      const z = features.reduce((sum, feature, idx) => sum + feature * (w[idx] || 0), bias);
      confidence = 1 / (1 + Math.exp(-z));
      prediction = confidence > 0.5 ? 1 : 0;
    } else {
      // Fallback simple
      prediction = Math.random() > 0.5 ? 1 : 0;
      confidence = 0.5;
    }

    return { prediction, confidence: Math.abs(confidence) };
  }

  // Entraîner un modèle
  static async trainModel(
    data: Review[],
    config: TrainingConfig,
    onProgress?: (progress: number, status: string) => void
  ): Promise<TrainedModel> {
    const startTime = Date.now();
    
    onProgress?.(0, 'Préparation des données...');
    
    // Préparer les données
    const texts = data.map(review => review.text);
    const labels = data.map(review => review.label);
    
    // Mélanger les données
    const shuffled = data.map((item, index) => ({ item, index }))
      .sort(() => Math.random() - 0.5);
    
    const shuffledTexts = shuffled.map(s => texts[s.index]);
    const shuffledLabels = shuffled.map(s => labels[s.index]);
    
    // Division train/test
    const splitIndex = Math.floor(shuffledTexts.length * (1 - config.testSize));
    const trainTexts = shuffledTexts.slice(0, splitIndex);
    const trainLabels = shuffledLabels.slice(0, splitIndex);
    const testTexts = shuffledTexts.slice(splitIndex);
    const testLabels = shuffledLabels.slice(splitIndex);
    
    onProgress?.(20, 'Extraction des features...');
    
    // Extraction des features
    const { features: trainFeatures, vocabulary } = this.extractFeatures(
      trainTexts,
      config.maxFeatures,
      config.ngrams
    );
    
    const { features: testFeatures } = this.extractFeatures(
      testTexts,
      config.maxFeatures,
      config.ngrams
    );
    
    onProgress?.(50, `Entraînement du modèle ${config.modelType}...`);
    
    // Entraînement selon le type de modèle
    let modelWeights: number[] | number[][];
    
    if (config.modelType === 'naive_bayes') {
      const nbModel = this.trainNaiveBayes(trainFeatures, trainLabels);
      modelWeights = [
        nbModel.classPriors,
        nbModel.featureMeans.flat(),
        nbModel.featureVars.flat()
      ];
    } else if (config.modelType === 'logistic_regression') {
      const lrModel = this.trainLogisticRegression(
        trainFeatures,
        trainLabels,
        config.epochs || 100,
        config.learningRate || 0.01
      );
      modelWeights = [...lrModel.weights, lrModel.bias];
    } else {
      // Modèle par défaut
      modelWeights = new Array(vocabulary.length).fill(0).map(() => Math.random());
    }
    
    onProgress?.(80, 'Évaluation du modèle...');
    
    // Créer le modèle temporaire pour les prédictions
    const tempModel: TrainedModel = {
      id: Date.now().toString(),
      name: `${config.modelType}_${Date.now()}`,
      type: config.modelType,
      config,
      vocabulary,
      weights: modelWeights,
      createdAt: new Date().toISOString(),
      trainedOn: trainTexts.length,
      metrics: {
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1Score: 0,
        confusionMatrix: [[0, 0], [0, 0]],
        trainingTime: 0,
        predictions: []
      }
    };
    
    // Évaluation
    const predictions: Array<{ text: string; predicted: number; actual: number; confidence: number }> = [];
    let correct = 0;
    const confusionMatrix = [[0, 0], [0, 0]];
    
    testTexts.forEach((text, idx) => {
      const { prediction, confidence } = this.predict(tempModel, text);
      const actual = testLabels[idx];
      
      predictions.push({ text, predicted: prediction, actual, confidence });
      
      if (prediction === actual) correct++;
      confusionMatrix[actual][prediction]++;
    });
    
    const accuracy = correct / testTexts.length;
    const precision = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]) || 0;
    const recall = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[1][0]) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    
    const trainingTime = Date.now() - startTime;
    
    onProgress?.(100, 'Modèle entraîné avec succès !');
    
    const finalModel: TrainedModel = {
      ...tempModel,
      metrics: {
        accuracy,
        precision,
        recall,
        f1Score,
        confusionMatrix,
        trainingTime,
        predictions: predictions.slice(0, 50) // Garder seulement 50 exemples
      }
    };
    
    // Sauvegarder le modèle
    this.models.push(finalModel);
    
    return finalModel;
  }

  // Obtenir tous les modèles entraînés
  static getModels(): TrainedModel[] {
    return this.models;
  }

  // Supprimer un modèle
  static deleteModel(modelId: string): boolean {
    const index = this.models.findIndex(m => m.id === modelId);
    if (index !== -1) {
      this.models.splice(index, 1);
      return true;
    }
    return false;
  }

  // Sauvegarder un modèle en JSON
  static downloadModel(model: TrainedModel): void {
    const blob = new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${model.name}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Charger un modèle depuis JSON
  static loadModel(jsonFile: File): Promise<TrainedModel> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const model = JSON.parse(e.target?.result as string) as TrainedModel;
          this.models.push(model);
          resolve(model);
        } catch (error) {
          reject(error);
        }
      };
      reader.readAsText(jsonFile);
    });
  }
} 