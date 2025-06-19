import React from 'react';
import { X, Copy, Download, Maximize2, Minimize2, Code, Terminal, FileText } from 'lucide-react';

interface CodePopupProps {
  isOpen: boolean;
  onClose: () => void;
  stepTitle: string;
  code: string;
  language: string;
  model: 'nltk' | 'bert';
  stepNumber: number;
}

export const CodePopup: React.FC<CodePopupProps> = ({
  isOpen,
  onClose,
  stepTitle,
  code,
  language,
  model,
  stepNumber
}) => {
  const [isMaximized, setIsMaximized] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState<'code' | 'explanation' | 'output'>('code');

  // Bloquer le scroll de la page quand la popup est ouverte
  React.useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    // Nettoyage quand le composant est démonté
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  // Fermer avec Escape
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    // Vous pourriez ajouter une notification toast ici
  };

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${stepTitle.toLowerCase().replace(/\s+/g, '-')}-${model}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const modelColors = {
    nltk: {
      primary: 'text-blue-400',
      bg: 'bg-blue-500/10',
      border: 'border-blue-500/30',
      gradient: 'from-blue-500/20 to-cyan-500/20'
    },
    bert: {
      primary: 'text-purple-400',
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      gradient: 'from-purple-500/20 to-pink-500/20'
    }
  };

  const colors = modelColors[model];

  const getExplanation = () => {
    const explanations = {
      nltk: {
        1: "Cette étape nettoie le texte en supprimant les caractères indésirables, convertit en minuscules et normalise les espaces. L'approche NLTK est basée sur des règles simples mais efficaces.",
        2: "La tokenisation NLTK divise le texte en mots en utilisant des séparateurs comme les espaces et la ponctuation. C'est une approche déterministe et rapide.",
        3: "Suppression des mots vides (stop words) en utilisant une liste prédéfinie. NLTK utilise une approche basée sur un dictionnaire statique.",
        4: "La lemmatisation NLTK réduit les mots à leur forme canonique en utilisant un dictionnaire de correspondances morphologiques."
      },
      bert: {
        1: "Le nettoyage pour BERT préserve certains caractères spéciaux importants pour la compréhension contextuelle. BERT est sensible à la casse et aux nuances.",
        2: "La tokenisation BERT utilise l'algorithme WordPiece qui décompose les mots en sous-unités, permettant de gérer les mots rares et les variations morphologiques.",
        3: "BERT utilise une approche contextuelle pour les stop words, analysant l'importance de chaque mot dans son contexte avant de décider de le supprimer.",
        4: "La lemmatisation BERT est contextuelle et utilise les embeddings pour comprendre les relations sémantiques entre les mots et leurs formes canoniques."
      }
    };

    return explanations[model][stepNumber as keyof typeof explanations[typeof model]] || "Explication non disponible.";
  };

  const getOutput = () => {
    const outputs = {
      nltk: {
        1: `# Sortie exemple pour le nettoyage NLTK
Texte original: "This product is AMAZING!!! I love it so much 😍"
Texte nettoyé: "this product is amazing i love it so much"

Statistiques:
- Caractères supprimés: 7 (ponctuation, emojis)
- Conversion en minuscules: Oui
- Espaces normalisés: Oui`,
        2: `# Sortie exemple pour la tokenisation NLTK
Texte: "this product is amazing i love it so much"
Tokens: ['this', 'product', 'is', 'amazing', 'i', 'love', 'it', 'so', 'much']

Statistiques:
- Nombre de tokens: 9
- Méthode: Séparation par espaces
- Filtrage longueur min: 2 caractères`,
        3: `# Sortie exemple pour les stop words NLTK
Tokens avant: ['this', 'product', 'is', 'amazing', 'i', 'love', 'it', 'so', 'much']
Tokens après: ['product', 'amazing', 'love', 'much']
Stop words supprimés: ['this', 'is', 'i', 'it', 'so']

Statistiques:
- Tokens conservés: 4
- Tokens supprimés: 5
- Taux de réduction: 55%`,
        4: `# Sortie exemple pour la lemmatisation NLTK
Tokens avant: ['product', 'amazing', 'love', 'much']
Tokens après: ['product', 'amaze', 'love', 'much']
Modifications: ['amazing' → 'amaze']

Statistiques:
- Tokens modifiés: 1
- Tokens inchangés: 3
- Formes canoniques trouvées: 1`
      },
      bert: {
        1: `# Sortie exemple pour le nettoyage BERT
Texte original: "This product is AMAZING!!! I love it so much 😍"
Texte nettoyé: "This product is AMAZING! I love it so much [EMO]"

Statistiques:
- Préservation de la casse: Oui
- Emojis remplacés par tokens spéciaux: 1
- Ponctuation normalisée: Oui`,
        2: `# Sortie exemple pour la tokenisation BERT WordPiece
Texte: "This product is AMAZING! I love it so much"
Tokens: ['This', 'product', 'is', 'AM', '##AZ', '##ING', '!', 'I', 'love', 'it', 'so', 'much']

Statistiques:
- Nombre de tokens: 12
- Sous-mots créés: 3 (AM, ##AZ, ##ING)
- Tokens spéciaux: 1 (!)`,
        3: `# Sortie exemple pour les stop words BERT contextuel
Tokens avant: ['This', 'product', 'is', 'AM', '##AZ', '##ING', '!', 'I', 'love', 'it', 'so', 'much']
Tokens après: ['This', 'product', 'AM', '##AZ', '##ING', '!', 'love', 'much']
Stop words supprimés: ['is', 'I', 'it', 'so']

Statistiques:
- Analyse contextuelle: Oui
- Préservation de "This" (début de phrase): Oui
- Tokens conservés: 8`,
        4: `# Sortie exemple pour la lemmatisation BERT contextuelle
Tokens avant: ['This', 'product', 'AM', '##AZ', '##ING', '!', 'love', 'much']
Tokens après: ['This', 'product', 'amaze', '!', 'love', 'much']
Modifications: ['AM##AZ##ING' → 'amaze']

Statistiques:
- Reconstruction de sous-mots: 1
- Lemmatisation contextuelle: Oui
- Embeddings utilisés: Oui`
      }
    };

    return outputs[model][stepNumber as keyof typeof outputs[typeof model]] || "Sortie non disponible.";
  };

  // Empêcher la propagation du clic pour éviter de fermer la popup
  const handlePopupClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  // Fermer la popup si on clique sur le backdrop
  const handleBackdropClick = () => {
    onClose();
  };

  return (
    <div 
      className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 overflow-hidden"
      onClick={handleBackdropClick}
    >
      <div 
        className={`bg-slate-900 rounded-2xl border border-white/20 shadow-2xl transition-all duration-300 flex flex-col ${
          isMaximized ? 'w-full h-full' : 'w-full max-w-6xl h-5/6'
        }`}
        onClick={handlePopupClick}
      >
        {/* En-tête - Fixe */}
        <div className={`flex items-center justify-between p-6 border-b border-white/10 bg-gradient-to-r ${colors.gradient} flex-shrink-0`}>
          <div className="flex items-center space-x-4">
            <div className={`${colors.bg} ${colors.border} border p-3 rounded-xl`}>
              <Code className={`h-6 w-6 ${colors.primary}`} />
            </div>
            <div>
              <h2 className="text-white font-bold text-xl">{stepTitle}</h2>
              <p className="text-white/70">
                {language} • {model.toUpperCase()} • Étape {stepNumber}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={copyCode}
              className={`p-2 ${colors.bg} ${colors.primary} rounded-lg hover:bg-opacity-30 transition-colors`}
              title="Copier le code"
            >
              <Copy className="h-5 w-5" />
            </button>
            
            <button
              onClick={downloadCode}
              className={`p-2 ${colors.bg} ${colors.primary} rounded-lg hover:bg-opacity-30 transition-colors`}
              title="Télécharger"
            >
              <Download className="h-5 w-5" />
            </button>
            
            <button
              onClick={() => setIsMaximized(!isMaximized)}
              className="p-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
              title={isMaximized ? "Réduire" : "Agrandir"}
            >
              {isMaximized ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
            </button>
            
            <button
              onClick={onClose}
              className="p-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
              title="Fermer (Échap)"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Onglets - Fixe */}
        <div className="flex border-b border-white/10 flex-shrink-0">
          {[
            { id: 'code', label: 'Code Source', icon: Code },
            { id: 'explanation', label: 'Explication', icon: FileText },
            { id: 'output', label: 'Sortie', icon: Terminal }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-6 py-4 transition-all ${
                activeTab === tab.id
                  ? `${colors.bg} ${colors.primary} border-b-2 ${colors.border.replace('border-', 'border-b-')}`
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Contenu - Scrollable */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'code' && (
            <div className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
              <pre className="p-6 text-sm font-mono text-white/90 whitespace-pre-wrap leading-relaxed">
                <code className={`language-${language}`}>
                  {code}
                </code>
              </pre>
            </div>
          )}

          {activeTab === 'explanation' && (
            <div className="p-6 h-full overflow-y-auto scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
              <div className="prose prose-invert max-w-none">
                <h3 className={`${colors.primary} font-bold text-lg mb-4`}>
                  Comment fonctionne cette étape ?
                </h3>
                <p className="text-white/80 text-base leading-relaxed mb-6">
                  {getExplanation()}
                </p>
                
                <h4 className="text-white font-medium text-md mb-3">Caractéristiques clés :</h4>
                <ul className="text-white/70 space-y-2">
                  {model === 'nltk' ? (
                    <>
                      <li>• Approche basée sur des règles linguistiques</li>
                      <li>• Traitement déterministe et reproductible</li>
                      <li>• Optimisé pour la vitesse d'exécution</li>
                      <li>• Utilise des dictionnaires pré-construits</li>
                    </>
                  ) : (
                    <>
                      <li>• Analyse contextuelle avec transformers</li>
                      <li>• Compréhension sémantique avancée</li>
                      <li>• Gestion des nuances linguistiques</li>
                      <li>• Utilise des embeddings pré-entraînés</li>
                    </>
                  )}
                </ul>

                <h4 className="text-white font-medium text-md mb-3 mt-6">Avantages de cette approche :</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className={`${colors.bg} p-4 rounded-lg ${colors.border} border`}>
                    <h5 className={`${colors.primary} font-medium mb-2`}>Performance</h5>
                    <p className="text-white/70 text-sm">
                      {model === 'nltk' 
                        ? 'Traitement rapide et efficace pour de gros volumes de données'
                        : 'Précision élevée grâce à l\'analyse contextuelle avancée'
                      }
                    </p>
                  </div>
                  <div className={`${colors.bg} p-4 rounded-lg ${colors.border} border`}>
                    <h5 className={`${colors.primary} font-medium mb-2`}>Fiabilité</h5>
                    <p className="text-white/70 text-sm">
                      {model === 'nltk' 
                        ? 'Résultats reproductibles et prévisibles'
                        : 'Gestion robuste des cas complexes et ambigus'
                      }
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'output' && (
            <div className="p-6 h-full overflow-y-auto scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
              <div className="bg-slate-800 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-white font-medium">Exemple de sortie</h4>
                  <span className={`px-2 py-1 ${colors.bg} ${colors.primary} rounded text-xs`}>
                    {model.toUpperCase()}
                  </span>
                </div>
                <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap leading-relaxed">
                  {getOutput()}
                </pre>
              </div>
              
              {/* Informations supplémentaires */}
              <div className="mt-6 grid md:grid-cols-2 gap-4">
                <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                  <h5 className="text-white font-medium mb-2">Métriques de performance</h5>
                  <ul className="text-white/70 text-sm space-y-1">
                    <li>• Temps d'exécution: {model === 'nltk' ? '~50ms' : '~200ms'}</li>
                    <li>• Mémoire utilisée: {model === 'nltk' ? '~10MB' : '~500MB'}</li>
                    <li>• Précision: {model === 'nltk' ? '85%' : '92%'}</li>
                  </ul>
                </div>
                <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                  <h5 className="text-white font-medium mb-2">Cas d'usage optimaux</h5>
                  <ul className="text-white/70 text-sm space-y-1">
                    {model === 'nltk' ? (
                      <>
                        <li>• Traitement de gros volumes</li>
                        <li>• Applications temps réel</li>
                        <li>• Ressources limitées</li>
                      </>
                    ) : (
                      <>
                        <li>• Analyse fine du sentiment</li>
                        <li>• Textes complexes</li>
                        <li>• Précision maximale</li>
                      </>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Pied de page - Fixe */}
        <div className="px-6 py-4 bg-white/5 border-t border-white/10 text-xs text-white/60 flex-shrink-0">
          <div className="flex justify-between items-center">
            <span>
              Lignes: {code.split('\n').length} • Caractères: {code.length} • Modèle: {model.toUpperCase()}
            </span>
            <span>
              Étape {stepNumber}/4 • {language} • Pipeline NLP • Appuyez sur Échap pour fermer
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};