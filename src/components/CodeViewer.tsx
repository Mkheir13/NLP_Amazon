import React from 'react';
import { Code, Copy, Eye, EyeOff, Download, Maximize2 } from 'lucide-react';

interface CodeViewerProps {
  stepTitle: string;
  code: string;
  language: string;
  model: 'nltk' | 'bert';
  isVisible: boolean;
  onToggle: () => void;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({
  stepTitle,
  code,
  language,
  model,
  isVisible,
  onToggle
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    // Vous pourriez ajouter une notification ici
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
      button: 'bg-blue-500/20 hover:bg-blue-500/30'
    },
    bert: {
      primary: 'text-purple-400',
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      button: 'bg-purple-500/20 hover:bg-purple-500/30'
    }
  };

  const colors = modelColors[model];

  if (!isVisible) return null;

  return (
    <div className={`mt-6 ${colors.bg} ${colors.border} border rounded-xl overflow-hidden`}>
      {/* En-tête */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <Code className={`h-5 w-5 ${colors.primary}`} />
          <div>
            <h4 className="text-white font-medium">Code Source - {stepTitle}</h4>
            <p className="text-white/60 text-sm">{language} • {model.toUpperCase()}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={copyCode}
            className={`p-2 ${colors.button} ${colors.primary} rounded-lg transition-colors`}
            title="Copier le code"
          >
            <Copy className="h-4 w-4" />
          </button>
          
          <button
            onClick={downloadCode}
            className={`p-2 ${colors.button} ${colors.primary} rounded-lg transition-colors`}
            title="Télécharger le code"
          >
            <Download className="h-4 w-4" />
          </button>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`p-2 ${colors.button} ${colors.primary} rounded-lg transition-colors`}
            title={isExpanded ? "Réduire" : "Agrandir"}
          >
            <Maximize2 className="h-4 w-4" />
          </button>
          
          <button
            onClick={onToggle}
            className="p-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
            title="Fermer"
          >
            <EyeOff className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Code */}
      <div className={`${isExpanded ? 'max-h-96' : 'max-h-64'} overflow-y-auto`}>
        <pre className="p-4 text-sm font-mono text-white/90 whitespace-pre-wrap">
          <code className={`language-${language}`}>
            {code}
          </code>
        </pre>
      </div>

      {/* Pied de page avec informations */}
      <div className="px-4 py-2 bg-white/5 border-t border-white/10 text-xs text-white/60">
        <div className="flex justify-between items-center">
          <span>Lignes: {code.split('\n').length} • Caractères: {code.length}</span>
          <span>Modèle: {model.toUpperCase()} • Langage: {language}</span>
        </div>
      </div>
    </div>
  );
};