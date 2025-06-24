// Configuration centralisée pour éliminer tous les éléments hardcodés
export interface APIConfig {
  baseUrl: string;
  endpoints: {
    embeddings: string;
    autoencoder: string;
    bert: string;
    sentiment: string;
    dataset: string;
  };
  timeout: number;
}

// Configuration simplifiée pour éviter les erreurs process.env
const getApiUrl = () => {
  // En développement, utiliser localhost
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'http://localhost:5000';
  }
  // En production, utiliser l'URL actuelle
  return typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:5000` : 'http://localhost:5000';
};

export interface ModelConfig {
  autoencoder: {
    defaultInputDim: number;
    defaultEncodingDim: number;
    defaultLearningRate: number;
    defaultEpochs: number;
    defaultBatchSize: number;
    hiddenLayers: number[];
    maxInputDim: number;
    minEncodingDim: number;
    maxEncodingDim: number;
  };
  embeddings: {
    defaultMaxFeatures: number;
    defaultNgrams: [number, number];
    defaultMinDf: number;
    defaultMaxDf: number;
  };
  bert: {
    availableModels: string[];
    defaultModel: string;
    defaultEpochs: number;
    defaultBatchSize: number;
    defaultLearningRate: number;
  };
  pipeline: {
    defaultMinTokenLength: number;
    defaultMaxTokens: number;
    defaultStopWords: string[];
  };
}

export interface UIConfig {
  theme: {
    colors: {
      primary: string;
      secondary: string;
      success: string;
      warning: string;
      error: string;
      info: string;
    };
    gradients: {
      primary: string;
      secondary: string;
      success: string;
      warning: string;
      error: string;
    };
  };
  animations: {
    defaultDuration: number;
    hoverScale: number;
    pulseOpacity: number;
  };
  layout: {
    maxWidth: string;
    borderRadius: string;
    spacing: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
  };
  search: {
    defaultResultsLimit: number;
    maxResultsLimit: number;
    similarityThresholds: {
      high: number;
      medium: number;
      low: number;
    };
  };
  sentiment: {
    confidenceThresholds: {
      high: number;
      medium: number;
      low: number;
    };
    emotions: {
      [key: string]: {
        emoji: string;
        color: string;
        bg: string;
        label: string;
      };
    };
  };
}

export interface DataConfig {
  dataset: {
    defaultSize: number;
    maxSize: number;
    minTextLength: number;
    categories: string[];
  };
  validation: {
    minTrainingTexts: number;
    maxTrainingTexts: number;
    testSizeRange: [number, number];
  };
}

// Configuration par défaut
export const DEFAULT_CONFIG = {
  api: {
    baseUrl: getApiUrl(),
    endpoints: {
      embeddings: '/api/embeddings',
      autoencoder: '/api/autoencoder',
      bert: '/api/bert',
      sentiment: '/api/sentiment',
      dataset: '/api/dataset'
    },
    timeout: 30000
  } as APIConfig,

  models: {
    autoencoder: {
      defaultInputDim: 1000,
      defaultEncodingDim: 64,
      defaultLearningRate: 0.0005,
      defaultEpochs: 100,
      defaultBatchSize: 16,
      hiddenLayers: [512, 128],
      maxInputDim: 5000,
      minEncodingDim: 16,
      maxEncodingDim: 512
    },
    embeddings: {
      defaultMaxFeatures: 1000,
      defaultNgrams: [1, 2] as [number, number],
      defaultMinDf: 2,
      defaultMaxDf: 0.8
    },
    bert: {
      availableModels: [
        'distilbert-base-uncased',
        'bert-base-uncased',
        'roberta-base'
      ],
      defaultModel: 'distilbert-base-uncased',
      defaultEpochs: 3,
      defaultBatchSize: 16,
      defaultLearningRate: 2e-5
    },
    pipeline: {
      defaultMinTokenLength: 2,
      defaultMaxTokens: 1000,
      defaultStopWords: ['ok', 'well', 'um', 'uh']
    }
  } as ModelConfig,

  ui: {
    theme: {
      colors: {
        primary: 'cyan-400',
        secondary: 'purple-400',
        success: 'green-400',
        warning: 'yellow-400',
        error: 'red-400',
        info: 'blue-400'
      },
      gradients: {
        primary: 'from-cyan-500 to-blue-600',
        secondary: 'from-purple-500 to-pink-600',
        success: 'from-green-500 to-emerald-600',
        warning: 'from-yellow-500 to-orange-600',
        error: 'from-red-500 to-rose-600'
      }
    },
    animations: {
      defaultDuration: 300,
      hoverScale: 1.05,
      pulseOpacity: 0.5
    },
    layout: {
      maxWidth: '7xl',
      borderRadius: 'xl',
      spacing: {
        xs: '1',
        sm: '2',
        md: '4',
        lg: '6',
        xl: '8'
      }
    },
    search: {
      defaultResultsLimit: 5,
      maxResultsLimit: 50,
      similarityThresholds: {
        high: 0.8,
        medium: 0.6,
        low: 0.4
      }
    },
    sentiment: {
      confidenceThresholds: {
        high: 0.8,
        medium: 0.6,
        low: 0.4
      },
      emotions: {
        joy: { emoji: '😊', color: 'text-yellow-400', bg: 'bg-yellow-500/20', label: 'Joie' },
        anger: { emoji: '😠', color: 'text-red-400', bg: 'bg-red-500/20', label: 'Colère' },
        fear: { emoji: '😨', color: 'text-purple-400', bg: 'bg-purple-500/20', label: 'Peur' },
        sadness: { emoji: '😢', color: 'text-blue-400', bg: 'bg-blue-500/20', label: 'Tristesse' },
        surprise: { emoji: '😲', color: 'text-orange-400', bg: 'bg-orange-500/20', label: 'Surprise' },
        disgust: { emoji: '🤢', color: 'text-green-400', bg: 'bg-green-500/20', label: 'Dégoût' }
      }
    }
  } as UIConfig,

  data: {
    dataset: {
      defaultSize: 1000,
      maxSize: 10000,
      minTextLength: 10,
      categories: ['positive', 'negative', 'neutral']
    },
    validation: {
      minTrainingTexts: 5,
      maxTrainingTexts: 1000,
      testSizeRange: [0.1, 0.5] as [number, number]
    }
  } as DataConfig
};

// Hook pour utiliser la configuration
export const useAppConfig = () => {
  return DEFAULT_CONFIG;
};

// Utilitaires pour la configuration
export class ConfigManager {
  private static config = DEFAULT_CONFIG;

  static getApiUrl(endpoint: keyof APIConfig['endpoints']): string {
    return `${this.config.api.baseUrl}${this.config.api.endpoints[endpoint]}`;
  }

  static getModelDefaults(model: keyof ModelConfig): any {
    return this.config.models[model];
  }

  static getUIConfig(): UIConfig {
    return this.config.ui;
  }

  static getDataConfig(): DataConfig {
    return this.config.data;
  }

  static updateConfig(newConfig: Partial<typeof DEFAULT_CONFIG>): void {
    this.config = { ...this.config, ...newConfig };
  }

  static resetToDefaults(): void {
    this.config = DEFAULT_CONFIG;
  }

  // Validation des configurations
  static validateAutoencoderConfig(config: any): boolean {
    const defaults = this.config.models.autoencoder;
    return (
      config.input_dim >= defaults.minEncodingDim &&
      config.input_dim <= defaults.maxInputDim &&
      config.encoding_dim >= defaults.minEncodingDim &&
      config.encoding_dim <= defaults.maxEncodingDim &&
      config.learning_rate > 0 &&
      config.learning_rate < 1 &&
      config.epochs > 0 &&
      config.batch_size > 0
    );
  }

  static validateSearchConfig(config: any): boolean {
    const defaults = this.config.ui.search;
    return (
      config.limit >= 1 &&
      config.limit <= defaults.maxResultsLimit
    );
  }
}

export default ConfigManager; 