import React, { useState } from 'react';

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze the text.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center py-12 px-4">
      <header className="max-w-3xl w-full text-center mb-8">
        <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
          Fake News Detector <span className="text-blue-500">with XAI</span>
        </h1>
        <p className="mt-3 text-lg text-slate-400">
          Analyze news articles and inspect word-level explanations powered by BERT and LIME.
        </p>
      </header>

      <main className="max-w-3xl w-full space-y-6">
        {/* Input Box Area */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
          <label htmlFor="article" className="block text-sm font-semibold text-slate-300 mb-2">
            Article Text or Headline
          </label>
          <textarea
            id="article"
            rows="6"
            className="w-full p-4 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition resize-none"
            placeholder="Paste news headline or full article text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className={`mt-4 w-full py-3 px-6 rounded-lg font-semibold text-white transition flex items-center justify-center space-x-2 ${
              loading || !text.trim()
                ? 'bg-slate-700 cursor-not-allowed opacity-60'
                : 'bg-blue-600 hover:bg-blue-500 cursor-pointer shadow-lg shadow-blue-500/30'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                </svg>
                <span>Analyzing with BERT & LIME...</span>
              </>
            ) : (
              <span>Analyze Article</span>
            )}
          </button>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="p-4 bg-red-950/80 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl space-y-6">
            {/* Classification & Confidence Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 bg-slate-900/80 rounded-lg border border-slate-700/60 gap-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-slate-400 font-bold">Prediction Result</span>
                <div className="flex items-center space-x-3 mt-1">
                  <span
                    className={`text-2xl font-black px-3 py-1 rounded-md ${
                      result.prediction === 'Real'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                    }`}
                  >
                    {result.prediction.toUpperCase()}
                  </span>
                  <span className="text-slate-300 text-sm font-medium">
                    Confidence: <strong>{result.confidence}%</strong>
                  </span>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="w-full sm:w-48 bg-slate-800 h-3 rounded-full overflow-hidden border border-slate-700">
                <div
                  className={`h-full transition-all duration-500 ${
                    result.prediction === 'Real' ? 'bg-emerald-500' : 'bg-rose-500'
                  }`}
                  style={{ width: `${result.confidence}%` }}
                />
              </div>
            </div>

            {/* Explainable AI Word Weights Section */}
            <div>
              <h2 className="text-xl font-bold text-slate-100">Explainable AI (LIME Analysis)</h2>
              <p className="text-sm text-slate-400 mt-1">
                Word importance impact on final prediction label:
              </p>

              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                {result.explanation.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-3 bg-slate-900 border border-slate-700/60 rounded-lg"
                  >
                    <span className="font-mono text-slate-200">{item.word}</span>
                    <span
                      className={`text-sm font-semibold font-mono ${
                        item.score > 0 ? 'text-emerald-400' : item.score < 0 ? 'text-rose-400' : 'text-slate-400'
                      }`}
                    >
                      {item.score > 0 ? `+${item.score}` : item.score}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}