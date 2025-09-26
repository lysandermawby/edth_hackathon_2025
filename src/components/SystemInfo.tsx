import React, { useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { OutputProps } from '../types'

const SystemInfo: React.FC<OutputProps> = ({ addToOutput }) => {
  const [systemInfo, setSystemInfo] = useState('Click refresh to load system information...')
  const [isLoading, setIsLoading] = useState(false)

  const refreshSystemInfo = async (): Promise<void> => {
    setIsLoading(true)

    try {
      const info = await invoke<string>('get_system_info')
      setSystemInfo(info)
      addToOutput('System information refreshed successfully')
    } catch (error) {
      const errorMessage = error as string
      setSystemInfo(`Error: ${errorMessage}`)
      addToOutput(`Failed to get system info: ${errorMessage}`, true)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="bg-white rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-200">
      <h2 className="text-2xl font-semibold text-gray-700 mb-5 border-b-2 border-primary-500 pb-3">
        System Information
      </h2>
      <div>
        <button
          onClick={refreshSystemInfo}
          disabled={isLoading}
          className="mb-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white px-6 py-2 rounded-lg hover:from-primary-600 hover:to-primary-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Loading...' : 'Refresh System Info'}
        </button>
        <pre className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm font-mono text-gray-700 overflow-auto">
          {systemInfo}
        </pre>
      </div>
    </section>
  )
}

export default SystemInfo