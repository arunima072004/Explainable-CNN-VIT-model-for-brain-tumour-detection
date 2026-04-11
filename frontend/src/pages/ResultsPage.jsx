import { useState } from 'react'
import axios from 'axios'
import ConfidenceBar from '../components/ConfidenceBar'
import ImageViewer from '../components/ImageViewer'
import TumorTypeBadge from '../components/TumorTypeBadge'

const VIEWS = ['original', 'heatmap', 'mask', 'overlay']

const VIEW_LABELS = {
  original: 'Original',
  heatmap: 'Grad-CAM',
  mask: 'Mask',
  overlay: 'Overlay',
}

export default function ResultsPage({ result, onReset, file, threshold }) {
  const [activeView, setActiveView] = useState('original')
  const [downloading, setDownloading] = useState(false)

  const hasTumor = result.tumor_detected
  const confidence = result.confidence
  const tumorType = result.tumor_type
  const typeConf = result.type_confidence
  const allProbs = result.all_probs || {}
  const bbox = result.bbox

  const images = {
    original: result.original,
    heatmap: result.heatmap,
    mask: result.mask,
    overlay: result.overlay,
  }

  const availableViews = VIEWS.filter(v => images[v])

  const handleDownloadPDF = async () => {
    setDownloading(true)
    try {
      const resp = await axios.post('/report',
        { result, filename: file ? file.name : 'scan.jpg' },
        { responseType: 'blob', headers: { 'Content-Type': 'application/json' } }
      )

      // Check if response is actually an error JSON blob
      const contentType = resp.headers['content-type'] || ''
      if (!contentType.includes('pdf')) {
        // Read the blob as text to get the error message
        const text = await resp.data.text()
        throw new Error(text)
      }

      const url  = URL.createObjectURL(new Blob([resp.data], { type: 'application/pdf' }))
      const link = document.createElement('a')
      link.href  = url
      link.download = `brain_tumor_report_${(file?.name || 'scan').replace(/\.[^.]+$/, '')}.pdf`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) {
      // If error response is a blob, read it as text
      let msg = e.message
      if (e.response?.data instanceof Blob) {
        msg = await e.response.data.text()
      }
      console.error('PDF error:', msg)
      alert('PDF generation failed: ' + msg)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">Analysis Results</h2>
        <div className="flex gap-2">
          <button
            onClick={handleDownloadPDF}
            disabled={downloading}
            className="text-sm text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-40
              disabled:cursor-not-allowed border border-blue-500 px-4 py-1.5 rounded-lg
              transition-colors flex items-center gap-2"
          >
            {downloading ? (
              <>
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Generating...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
                </svg>
                Download PDF
              </>
            )}
          </button>
          <button
            onClick={onReset}
            className="text-sm text-gray-400 hover:text-white border border-gray-700
              hover:border-gray-500 px-4 py-1.5 rounded-lg transition-colors"
          >
            ← New Scan
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Image viewer */}
        <div className="flex flex-col gap-3">
          {/* View toggle */}
          <div className="flex gap-2 flex-wrap">
            {availableViews.map(v => (
              <button
                key={v}
                onClick={() => setActiveView(v)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                  ${activeView === v
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'}`}
              >
                {VIEW_LABELS[v]}
              </button>
            ))}
          </div>

          <ImageViewer
            src={images[activeView]}
            label={VIEW_LABELS[activeView]}
            bbox={activeView === 'original' && hasTumor ? bbox : null}
          />
        </div>

        {/* Right: Results panel */}
        <div className="flex flex-col gap-4">
          {/* Detection result */}
          <div className={`rounded-xl p-5 border ${hasTumor
            ? 'bg-red-950/40 border-red-800'
            : 'bg-green-950/40 border-green-800'}`}>
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-3 h-3 rounded-full ${hasTumor ? 'bg-red-400' : 'bg-green-400'}`} />
              <span className="font-semibold text-lg">
                {hasTumor ? 'Tumor Detected' : 'No Tumor Detected'}
              </span>
            </div>
            <ConfidenceBar
              label="Detection Confidence"
              value={hasTumor ? confidence : 1 - confidence}
              color={hasTumor ? 'red' : 'green'}
            />
          </div>

          {/* Tumor type */}
          {hasTumor && (
            <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Tumor Type</span>
                <TumorTypeBadge type={tumorType} />
              </div>

              <ConfidenceBar
                label="Classification Confidence"
                value={typeConf}
                color="blue"
              />

              {/* All class probabilities */}
              {Object.keys(allProbs).length > 0 && (
                <div className="flex flex-col gap-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wide">All Classes</span>
                  {Object.entries(allProbs).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-36 truncate">
                        {cls.replace('_tumor', '').replace('_', ' ')}
                      </span>
                      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
                        <div
                          className="bg-blue-500 h-1.5 rounded-full transition-all"
                          style={{ width: `${(prob * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-10 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Bounding box info */}
              {bbox && (
                <div className="text-xs text-gray-500 border-t border-gray-800 pt-3">
                  Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}
                </div>
              )}
            </div>
          )}

          {/* Clinical note */}
          <ClinicalNote tumorType={tumorType} hasTumor={hasTumor} />
        </div>
      </div>
    </div>
  )
}

function ClinicalNote({ tumorType, hasTumor }) {
  const notes = {
    glioma_tumor: 'Gliomas arise from glial cells and are the most common primary brain tumors. They vary widely in aggressiveness.',
    meningioma_tumor: 'Meningiomas develop from the meninges. Usually benign and slow-growing, representing ~36% of all primary brain tumors.',
    pituitary_tumor: 'Pituitary tumors affect the pituitary gland and can disrupt hormone regulation. Most are benign and treatable.',
  }

  const note = hasTumor ? notes[tumorType] : 'No abnormal tissue patterns were detected in this scan.'

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Clinical Note</p>
      <p className="text-sm text-gray-300 leading-relaxed">{note}</p>
      <p className="text-xs text-gray-600 mt-3">
        ⚠ For research purposes only. Not a substitute for professional medical diagnosis.
      </p>
    </div>
  )
}
