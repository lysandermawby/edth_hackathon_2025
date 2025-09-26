import React, { useState, useEffect } from 'react'
import SystemInfo from './components/SystemInfo'
import TrackingSection from './components/TrackingSection'
import VideoSection from './components/VideoSection'
import OutputSection from './components/OutputSection'

function App(): React.ReactElement {
  const [output, setOutput] = useState('Ready to run tracking operations...\n')

  const addToOutput = (message: string, isError = false): void => {
    const timestamp = new Date().toLocaleTimeString()
    const prefix = isError ? '[ERROR]' : '[INFO]'
    const formattedMessage = `${timestamp} ${prefix} ${message}\n`

    setOutput(prev => prev + formattedMessage)
  }

  const clearOutput = (): void => {
    setOutput('Output cleared...\n')
  }

  useEffect(() => {
    addToOutput('EDTH Object Tracker GUI initialized')
    addToOutput('Click "Refresh System Info" to check Python environment')
    addToOutput('Click "Refresh Video List" to see available videos in data/ directory')
  }, [])

  return (
    <div className="bg-gradient-to-br from-primary-500 to-secondary-500 min-h-screen">
      <div className="max-w-7xl mx-auto p-6">
        <header className="text-center mb-8 text-white">
          <h1 className="text-4xl font-bold mb-3 drop-shadow-lg">EDTH Object Tracker</h1>
          <p className="text-lg opacity-90">Real-time object tracking and video processing GUI</p>
        </header>

        <div className="space-y-6">
          <SystemInfo addToOutput={addToOutput} />
          <TrackingSection addToOutput={addToOutput} />
          <VideoSection addToOutput={addToOutput} />
          <OutputSection output={output} onClear={clearOutput} />
        </div>
      </div>
    </div>
  )
}

export default App