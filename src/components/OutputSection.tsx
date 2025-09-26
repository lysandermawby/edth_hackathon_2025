import React, { useEffect, useRef } from 'react'
import { OutputSectionProps } from '../types'

const OutputSection: React.FC<OutputSectionProps> = ({ output, onClear }) => {
  const outputRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  return (
    <section className="bg-white rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-200">
      <h2 className="text-2xl font-semibold text-gray-700 mb-5 border-b-2 border-primary-500 pb-3">
        Output
      </h2>
      <div className="mb-4">
        <button
          onClick={onClear}
          className="bg-gradient-to-r from-red-500 to-red-600 text-white px-6 py-2 rounded-lg hover:from-red-600 hover:to-red-700 transition-all duration-200 shadow-md hover:shadow-lg"
        >
          Clear Output
        </button>
      </div>
      <pre
        ref={outputRef}
        className="bg-gray-900 text-green-400 rounded-lg p-5 text-sm font-mono max-h-96 overflow-y-auto border border-gray-700"
      >
        {output}
      </pre>
    </section>
  )
}

export default OutputSection