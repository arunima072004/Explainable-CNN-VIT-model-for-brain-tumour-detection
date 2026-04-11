import { useRef, useState, useEffect } from 'react'

/**
 * ImageViewer - displays a base64 image.
 * When bbox is provided, draws a red bounding box using a canvas overlay
 * that matches the rendered image dimensions.
 */
export default function ImageViewer({ src, label, bbox }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)

  // Draw bbox on canvas once image is loaded
  useEffect(() => {
    if (!bbox || !imgLoaded || !imgRef.current || !canvasRef.current) return

    const img = imgRef.current
    const canvas = canvasRef.current

    // Match canvas size to rendered image
    const rect = img.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Scale bbox from natural image coords to rendered coords
    const scaleX = rect.width / img.naturalWidth
    const scaleY = rect.height / img.naturalHeight

    const [x, y, w, h] = bbox
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY)
  }, [bbox, imgLoaded, src])

  if (!src) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center h-64 text-gray-600 text-sm">
        No image available
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      <div className="relative flex justify-center">
        <img
          ref={imgRef}
          src={src}
          alt={label}
          className="w-full object-contain max-h-80"
          onLoad={() => setImgLoaded(true)}
        />
        {bbox && (
          <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </div>
      <div className="px-3 py-2 text-xs text-gray-500 border-t border-gray-800">
        {label}
      </div>
    </div>
  )
}
