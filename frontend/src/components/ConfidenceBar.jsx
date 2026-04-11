export default function ConfidenceBar({ label, value, color = 'blue' }) {
  const pct = (value * 100).toFixed(1)

  const barColor = {
    blue:  'bg-blue-500',
    red:   'bg-red-500',
    green: 'bg-green-500',
  }[color] || 'bg-blue-500'

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono font-semibold text-white">{pct}%</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div
          className={`${barColor} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
