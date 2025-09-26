import React, { useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { OutputProps } from '../types'

interface TrackerOption {
  id: string
  title: string
  description: string
  type: 'yolo' | 'transformer' | 'run_menu'
}

const trackerOptions: TrackerOption[] = [
  {
    id: 'yolo',
    title: 'YOLO Tracker',
    description: 'Fast real-time tracking using YOLO model',
    type: 'yolo'
  },
  {
    id: 'transformer',
    title: 'Transformer Tracker',
    description: 'More accurate tracking using RT-DETR transformer model',
    type: 'transformer'
  },
  {
    id: 'menu',
    title: 'Interactive Menu',
    description: 'Choose tracker interactively via command line',
    type: 'run_menu'
  }
]

const TrackingSection: React.FC<OutputProps> = ({ addToOutput }) => {
  const [loadingTracker, setLoadingTracker] = useState<string | null>(null)

  const startTracker = async (trackerType: string, buttonId: string): Promise<void> => {
    setLoadingTracker(buttonId)
    addToOutput(`Starting ${trackerType} tracker...`)

    try {
      const result = await invoke<string>('run_python_tracker', { trackerType })
      addToOutput(`${trackerType} tracker output:\n${result}`)
    } catch (error) {
      const errorMessage = error as string
      addToOutput(`Failed to start ${trackerType} tracker: ${errorMessage}`, true)
    } finally {
      setLoadingTracker(null)
    }
  }

  return (
    <section className="bg-white rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-200">
      <h2 className="text-2xl font-semibold text-gray-700 mb-5 border-b-2 border-primary-500 pb-3">
        Real-time Tracking
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {trackerOptions.map((option) => (
          <div
            key={option.id}
            className="border border-gray-200 rounded-lg p-5 text-center bg-gray-50 hover:bg-primary-50 hover:border-primary-300 transition-all duration-200"
          >
            <h3 className="text-xl font-semibold text-gray-800 mb-2">{option.title}</h3>
            <p className="text-gray-600 text-sm mb-4">{option.description}</p>
            <button
              onClick={() => startTracker(option.type, option.id)}
              disabled={loadingTracker === option.id}
              className="w-full bg-gradient-to-r from-primary-500 to-primary-600 text-white px-4 py-2 rounded-lg hover:from-primary-600 hover:to-primary-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadingTracker === option.id ? 'Loading...' : `Start ${option.title}`}
            </button>
          </div>
        ))}
      </div>
    </section>
  )
}

export default TrackingSection