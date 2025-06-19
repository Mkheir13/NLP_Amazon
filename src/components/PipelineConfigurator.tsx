import React from 'react';
import { Settings, Save, RotateCcw, Download, Upload, Sliders, ToggleLeft, ToggleRight } from 'lucide-react';

interface PipelineConfig {
  steps: {
    cleaning: boolean;
    tokenization: boolean;
    stopwords: boolean;
    lemmatization: boolean;
  };
  parameters: {
    minTokenLength: number;
    maxTokens: number;
    customStopWords: string[];
    caseSensitive: boolean;
    preserveNumbers: boolean;
    preservePunctuation: boolean;
  };
  templates: {
    name: string;
    description: string;
  };
}

interface PipelineConfiguratorProps {
  config: PipelineConfig;
  onConfigChange: (config: PipelineConfig) => void;
  model: 'nltk' | 'bert';
}

export const PipelineConfigurator: React.FC<PipelineConfiguratorProps> = ({
  config,
  onConfigChange,
  model
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState<'steps' | 'parameters' | 'templates'>('steps');

  // Templates prédéfinis
  const predefinedTemplates = {
    'sentiment-analysis': {
      name: 'Analyse de Sentiment',
      description: 'Optimisé pour l\'analyse de sentiment des avis clients',
      config: {
        steps: { cleaning: true, tokenization: true, stopwords: true, lemmatization: true },
        parameters: {
          minTokenLength: 2,
          maxTokens: 1000,
          customStopWords: ['ok', 'well', 'um', 'uh'],
          caseSensitive: false,
          preserveNumbers: false,
          preservePunctuation: false
        }
      }
    },
    'keyword-extraction': {
      name: 'Extraction de Mots-clés',
      description: 'Conserve plus d\'informations pour identifier les mots-clés',
      config: {
        steps: { cleaning: true, tokenization: true, stopwords: false, lemmatization: true },
        parameters: {
          minTokenLength: 3,
          maxTokens: 500,
          customStopWords: [],
          caseSensitive: true,
          preserveNumbers: true,
          preservePunctuation: true
        }
      }
    },
    'minimal-processing': {
      name: 'Traitement Minimal',
      description: 'Traitement léger pour préserver le maximum de contexte',
      config: {
        steps: { cleaning: true, tokenization: true, stopwords: false, lemmatization: false },
        parameters: {
          minTokenLength: 1,
          maxTokens: 2000,
          customStopWords: [],
          caseSensitive: true,
          preserveNumbers: true,
          preservePunctuation: true
        }
      }
    }
  };

  const updateSteps = (step: keyof PipelineConfig['steps'], enabled: boolean) => {
    onConfigChange({
      ...config,
      steps: { ...config.steps, [step]: enabled }
    });
  };

  const updateParameters = (param: keyof PipelineConfig['parameters'], value: any) => {
    onConfigChange({
      ...config,
      parameters: { ...config.parameters, [param]: value }
    });
  };

  const applyTemplate = (templateKey: string) => {
    const template = predefinedTemplates[templateKey as keyof typeof predefinedTemplates];
    if (template) {
      onConfigChange({
        ...config,
        ...template.config,
        templates: { name: template.name, description: template.description }
      });
    }
  };

  const resetToDefault = () => {
    onConfigChange({
      steps: { cleaning: true, tokenization: true, stopwords: true, lemmatization: true },
      parameters: {
        minTokenLength: 2,
        maxTokens: 1000,
        customStopWords: [],
        caseSensitive: false,
        preserveNumbers: false,
        preservePunctuation: false
      },
      templates: { name: 'Configuration par défaut', description: 'Configuration standard' }
    });
  };

  const exportConfig = () => {
    const dataStr = JSON.stringify(config, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `pipeline-config-${model}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const importConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedConfig = JSON.parse(e.target?.result as string);
          onConfigChange(importedConfig);
        } catch (error) {
          alert('Erreur lors de l\'importation du fichier de configuration');
        }
      };
      reader.readAsText(file);
    }
  };

  const Toggle: React.FC<{ enabled: boolean; onChange: (enabled: boolean) => void; label: string }> = 
    ({ enabled, onChange, label }) => (
      <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
        <span className="text-white/80">{label}</span>
        <button
          onClick={() => onChange(!enabled)}
          className={`flex items-center transition-colors ${
            enabled ? 'text-green-400' : 'text-white/40'
          }`}
        >
          {enabled ? <ToggleRight className="h-6 w-6" /> : <ToggleLeft className="h-6 w-6" />}
        </button>
      </div>
    );

  return (
    <div className="space-y-4">
      {/* En-tête */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-4 rounded-xl border border-white/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Settings className="h-6 w-6 text-cyan-400" />
            <div>
              <h3 className="text-white font-bold text-lg">Configuration Pipeline {model.toUpperCase()}</h3>
              <p className="text-white/60 text-sm">{config.templates.description}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
            >
              <Sliders className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Configuration détaillée */}
      {isExpanded && (
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-6 rounded-xl border border-white/20">
          {/* Onglets */}
          <div className="flex space-x-2 mb-6">
            {[
              { id: 'steps', label: 'Étapes', icon: Settings },
              { id: 'parameters', label: 'Paramètres', icon: Sliders },
              { id: 'templates', label: 'Templates', icon: Save }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'bg-white/10 text-white/70 hover:text-white hover:bg-white/20'
                }`}
              >
                <tab.icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Contenu des onglets */}
          {activeTab === 'steps' && (
            <div className="space-y-4">
              <h4 className="text-white font-medium text-lg mb-4">Étapes du Pipeline</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <Toggle
                  enabled={config.steps.cleaning}
                  onChange={(enabled) => updateSteps('cleaning', enabled)}
                  label="Nettoyage du texte"
                />
                <Toggle
                  enabled={config.steps.tokenization}
                  onChange={(enabled) => updateSteps('tokenization', enabled)}
                  label="Tokenisation"
                />
                <Toggle
                  enabled={config.steps.stopwords}
                  onChange={(enabled) => updateSteps('stopwords', enabled)}
                  label="Suppression stop words"
                />
                <Toggle
                  enabled={config.steps.lemmatization}
                  onChange={(enabled) => updateSteps('lemmatization', enabled)}
                  label="Lemmatisation"
                />
              </div>
            </div>
          )}

          {activeTab === 'parameters' && (
            <div className="space-y-6">
              <h4 className="text-white font-medium text-lg mb-4">Paramètres Avancés</h4>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-white/80 text-sm mb-2">Longueur minimale des tokens</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={config.parameters.minTokenLength}
                    onChange={(e) => updateParameters('minTokenLength', parseInt(e.target.value))}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                  />
                </div>
                
                <div>
                  <label className="block text-white/80 text-sm mb-2">Nombre maximum de tokens</label>
                  <input
                    type="number"
                    min="100"
                    max="5000"
                    step="100"
                    value={config.parameters.maxTokens}
                    onChange={(e) => updateParameters('maxTokens', parseInt(e.target.value))}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                  />
                </div>
              </div>

              <div>
                <label className="block text-white/80 text-sm mb-2">Stop words personnalisés (séparés par des virgules)</label>
                <textarea
                  value={config.parameters.customStopWords.join(', ')}
                  onChange={(e) => updateParameters('customStopWords', 
                    e.target.value.split(',').map(w => w.trim()).filter(w => w.length > 0)
                  )}
                  className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-cyan-400 h-20 resize-none"
                  placeholder="mot1, mot2, mot3..."
                />
              </div>

              <div className="space-y-3">
                <Toggle
                  enabled={config.parameters.caseSensitive}
                  onChange={(enabled) => updateParameters('caseSensitive', enabled)}
                  label="Sensible à la casse"
                />
                <Toggle
                  enabled={config.parameters.preserveNumbers}
                  onChange={(enabled) => updateParameters('preserveNumbers', enabled)}
                  label="Préserver les nombres"
                />
                <Toggle
                  enabled={config.parameters.preservePunctuation}
                  onChange={(enabled) => updateParameters('preservePunctuation', enabled)}
                  label="Préserver la ponctuation"
                />
              </div>
            </div>
          )}

          {activeTab === 'templates' && (
            <div className="space-y-6">
              <h4 className="text-white font-medium text-lg mb-4">Templates Prédéfinis</h4>
              
              <div className="grid gap-4">
                {Object.entries(predefinedTemplates).map(([key, template]) => (
                  <div key={key} className="p-4 bg-white/5 rounded-lg border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="text-white font-medium">{template.name}</h5>
                      <button
                        onClick={() => applyTemplate(key)}
                        className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors text-sm"
                      >
                        Appliquer
                      </button>
                    </div>
                    <p className="text-white/70 text-sm">{template.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-wrap gap-3 mt-6 pt-6 border-t border-white/20">
            <button
              onClick={resetToDefault}
              className="flex items-center space-x-2 px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
              <span>Réinitialiser</span>
            </button>
            
            <button
              onClick={exportConfig}
              className="flex items-center space-x-2 px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors"
            >
              <Download className="h-4 w-4" />
              <span>Exporter</span>
            </button>
            
            <label className="flex items-center space-x-2 px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors cursor-pointer">
              <Upload className="h-4 w-4" />
              <span>Importer</span>
              <input
                type="file"
                accept=".json"
                onChange={importConfig}
                className="hidden"
              />
            </label>
          </div>
        </div>
      )}
    </div>
  );
};