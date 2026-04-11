const TYPE_CONFIG = {
  glioma_tumor:     { label: 'Glioma',     color: 'bg-orange-900 text-orange-300 border-orange-700' },
  meningioma_tumor: { label: 'Meningioma', color: 'bg-purple-900 text-purple-300 border-purple-700' },
  pituitary_tumor:  { label: 'Pituitary',  color: 'bg-yellow-900 text-yellow-300 border-yellow-700' },
}

export default function TumorTypeBadge({ type }) {
  const config = TYPE_CONFIG[type] || { label: type || 'N/A', color: 'bg-gray-800 text-gray-300 border-gray-700' }

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${config.color}`}>
      {config.label}
    </span>
  )
}
