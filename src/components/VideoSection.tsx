import React, { useState, useEffect } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { OutputProps } from '../types'

const VideoSection: React.FC<OutputProps> = ({ addToOutput }) => {
  const [videoFiles, setVideoFiles] = useState<string[]>([])
  const [selectedVideo, setSelectedVideo] = useState('')
  const [customVideoPath, setCustomVideoPath] = useState('')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const refreshVideoList = async (): Promise<void> => {
    setIsRefreshing(true)

    try {
      const files = await invoke<string[]>('list_video_files')
      setVideoFiles(files)

      if (files.length === 0) {
        addToOutput('No video files found in data/ directory')
      } else {
        addToOutput(`Found ${files.length} video file(s)`)
      }
    } catch (error) {
      const errorMessage = error as string
      addToOutput(`Failed to refresh video list: ${errorMessage}`, true)
    } finally {
      setIsRefreshing(false)
    }
  }

  const processVideo = async (videoPath: string): Promise<void> => {
    if (!videoPath || videoPath.trim() === '') {
      addToOutput('Please select or specify a video file', true)
      return
    }

    setIsProcessing(true)
    addToOutput(`Processing video: ${videoPath}`)

    try {
      const result = await invoke<string>('process_video', { videoPath })
      addToOutput(`Video processing completed:\n${result}`)
      addToOutput(`Output video should be saved as: ${videoPath.replace(/\.[^/.]+$/, '_output.mp4')}`)
    } catch (error) {
      const errorMessage = error as string
      addToOutput(`Failed to process video: ${errorMessage}`, true)
    } finally {
      setIsProcessing(false)
    }
  }

  useEffect(() => {
    refreshVideoList()
  }, [])

  return (
    <section className="bg-white rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-200">
      <h2 className="text-2xl font-semibold text-gray-700 mb-5 border-b-2 border-primary-500 pb-3">
        Video Processing
      </h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-4 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Available Video Files</h3>
          <button
            onClick={refreshVideoList}
            disabled={isRefreshing}
            className="mb-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white px-4 py-2 rounded-lg hover:from-primary-600 hover:to-primary-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRefreshing ? 'Loading...' : 'Refresh Video List'}
          </button>
          <select
            value={selectedVideo}
            onChange={(e) => setSelectedVideo(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="">Select a video file...</option>
            {videoFiles.map((file) => (
              <option key={file} value={file}>
                {file.replace('data/', '')}
              </option>
            ))}
          </select>
        </div>
        <div className="p-4 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Processing Controls</h3>
          <button
            onClick={() => processVideo(selectedVideo)}
            disabled={!selectedVideo || isProcessing}
            className="w-full mb-4 bg-gradient-to-r from-green-500 to-green-600 text-white px-4 py-2 rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed"
          >
            {isProcessing ? 'Processing...' : 'Process Selected Video'}
          </button>
          <div className="pt-4 border-t border-gray-200">
            <p className="text-sm text-gray-600 mb-2">Or specify a custom video path:</p>
            <input
              type="text"
              value={customVideoPath}
              onChange={(e) => setCustomVideoPath(e.target.value)}
              placeholder="data/your_video.mp4"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 mb-3"
            />
            <button
              onClick={() => processVideo(customVideoPath)}
              disabled={isProcessing}
              className="w-full bg-gradient-to-r from-green-500 to-green-600 text-white px-4 py-2 rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Processing...' : 'Process Custom Video'}
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

export default VideoSection