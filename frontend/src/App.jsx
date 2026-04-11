import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

export default function App() {
  const [result, setResult] = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadedThreshold, setUploadedThreshold] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleReset = () => {
    setResult(null)
    setError(null)
    setUploadedFile(null)
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm">
            AI
          </div>
          <h1 className="text-lg font-semibold text-white">Brain Tumor Detection</h1>
          <span className="text-xs text-gray-500 ml-1">ViT + ResNet-50 · XAI</span>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        {result ? (
          <ResultsPage
            result={result}
            file={uploadedFile}
            threshold={uploadedThreshold}
            onReset={handleReset}
          />
        ) : (
          <UploadPage
            loading={loading}
            error={error}
            setLoading={setLoading}
            setError={setError}
            setResult={setResult}
            setUploadedFile={setUploadedFile}
            setUploadedThreshold={setUploadedThreshold}
          />
        )}
      </main>
    </div>
  )
}
