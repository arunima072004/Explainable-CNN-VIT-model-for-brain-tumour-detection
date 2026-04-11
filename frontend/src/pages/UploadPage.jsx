import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

export default function UploadPage({ loading, error, setLoading, setError, setResult, setUploadedFile, setUploadedThreshold }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [threshold, setThreshold] = useState(0.5)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setError(null)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) handleFile(f)
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', file)
    formData.append('threshold', threshold)

    try {
      const { data } = await axios.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setUploadedFile(file)
      setUploadedThreshold(threshold)
      setResult(data)
    } catch (err) {
      const msg = err.response?.data?.error || 'Server error. Is the backend running?'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto mt-8 flex flex-col gap-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white">Upload MRI Scan</h2>
        <p className="text-gray-400 text-sm mt-1">
          Supports JPG, PNG · Analyzed by ViT + ResNet-50
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
          ${dragging ? 'border-blue-400 bg-blue-950/30' : 'border-gray-700 hover:border-gray-500'}
          ${preview ? 'border-blue-600' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {preview ? (
          <div className="flex flex-col items-center gap-3">
            <img src={preview} alt="preview" className="max-h-48 rounded-lg object-contain" />
            <span className="text-sm text-gray-400">{file.name}</span>
            <span className="text-xs text-blue-400">Click to change</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm">Drag & drop an MRI image here</p>
            <p className="text-xs">or click to browse</p>
          </div>
        )}
      </div>

      {/* Threshold slider */}
      <div className="bg-gray-900 rounded-xl p-4 flex flex-col gap-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-300">Masking Threshold</span>
          <span className="text-blue-400 font-mono">{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range" min="0.1" max="0.9" step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
        <p className="text-xs text-gray-500">
          Lower = larger mask region · Higher = more precise
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-950 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className="w-full py-3 rounded-xl font-semibold text-white bg-blue-600 hover:bg-blue-500
          disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Analyzing...
          </>
        ) : 'Analyze MRI'}
      </button>
    </div>
  )
}
