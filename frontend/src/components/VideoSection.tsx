import React, { useEffect, useState } from 'react'
import { invoke, convertFileSrc } from '@tauri-apps/api/core'
import AnnotatedVideoCanvas from './AnnotatedVideoCanvas'
import { OutputProps, FrameDetections } from '../types'

interface PreviewState {
  videoUrl: string
  absoluteVideoPath: string
  trackingPath: string
  trackingData: FrameDetections[]
  source: string
  detections: number
}

const normaliseVideoInput = (rawPath: string): string | null => {
  if (!rawPath) return null

  const trimmed = rawPath.trim()
  if (!trimmed) return null

  const isAbsolute = /^(?:[a-zA-Z]:[\\/]|\\\\|\/)/.test(trimmed)
  const normalisedSeparators = trimmed.replace(/\\/g, '/')

  if (isAbsolute) return normalisedSeparators

  const withoutLeadingDots = normalisedSeparators.replace(/^\.\/*/, '')

  if (withoutLeadingDots.startsWith('data/') || withoutLeadingDots.startsWith('../')) {
    return withoutLeadingDots
  }

  return `data/${withoutLeadingDots}`
}

const VideoSection: React.FC<OutputProps> = ({ addToOutput }) => {
  const [videoFiles, setVideoFiles] = useState<string[]>([])
  const [selectedVideo, setSelectedVideo] = useState('')
  const [customVideoPath, setCustomVideoPath] = useState('')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isBrowsing, setIsBrowsing] = useState(false)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [preview, setPreview] = useState<PreviewState | null>(null)

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

  const openFileDialog = async (): Promise<void> => {
    setIsBrowsing(true)

    try {
      const filePath = await invoke<string | null>('open_video_file_dialog')

      if (filePath) {
        setCustomVideoPath(filePath)
        addToOutput(`Selected video file: ${filePath}`)
      } else {
        addToOutput('File selection cancelled')
      }
    } catch (error) {
      const errorMessage = error as string
      addToOutput(`Failed to open file dialog: ${errorMessage}`, true)
    } finally {
      setIsBrowsing(false)
    }
  }

  const processVideo = async (rawPath: string): Promise<void> => {
    const resolvedPath = normaliseVideoInput(rawPath)

    if (!resolvedPath) {
      addToOutput('Please select or specify a video file', true)
      return
    }

    setIsProcessing(true)

    addToOutput(`Processing video: ${resolvedPath}`)

    try {
      const result = await invoke<string>('process_video', { videoPath: resolvedPath })
      addToOutput(`Video processing completed:\n${result}`)
      addToOutput(`Output video should be saved as: ${resolvedPath.replace(/\.[^/.]+$/, '_output.mp4')}`)
    } catch (error) {
      const errorMessage = error as string
      addToOutput(`Failed to process video: ${errorMessage}`, true)
    } finally {
      setIsProcessing(false)
    }
  }

  const previewVideo = async (rawPath: string): Promise<void> => {
    const resolvedPath = normaliseVideoInput(rawPath)

    if (!resolvedPath) {
      addToOutput('Please select or specify a video file', true)
      return
    }

    setIsPreviewLoading(true)
    setPreview(null)

    try {
      const absoluteVideoPath = await invoke<string>('resolve_media_path', { mediaPath: resolvedPath })
      const response = await invoke<{ content: string; path: string }>('load_tracking_data', {
        videoPath: resolvedPath
      })

      const parsed = JSON.parse(response.content) as FrameDetections[]
      const detections = parsed.reduce((total, frame) => total + frame.objects.length, 0)

      const videoUrl = convertFileSrc(absoluteVideoPath)

      setPreview({
        videoUrl,
        absoluteVideoPath,
        trackingPath: response.path,
        trackingData: parsed,
        source: resolvedPath,
        detections
      })

      addToOutput(`Preview ready for ${resolvedPath}. Loaded annotations from ${response.path}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      addToOutput(`Failed to load preview: ${message}`, true)
      setPreview(null)
    } finally {
      setIsPreviewLoading(false)
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
          <button
            onClick={() => previewVideo(selectedVideo)}
            disabled={!selectedVideo || isPreviewLoading}
            className="mt-4 w-full bg-gradient-to-r from-indigo-500 to-indigo-600 text-white px-4 py-2 rounded-lg hover:from-indigo-600 hover:to-indigo-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed"
          >
            {isPreviewLoading ? 'Loading Preview...' : 'Preview Selected Video'}
          </button>
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
            <p className="text-sm text-gray-600 mb-2">Or browse for a video file:</p>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={customVideoPath}
                onChange={(e) => setCustomVideoPath(e.target.value)}
                placeholder="Select or type video file path..."
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <button
                onClick={openFileDialog}
                disabled={isBrowsing}
                className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 py-3 rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
              >
                {isBrowsing ? 'Opening...' : '📁 Browse'}
              </button>
            </div>
            <button
              onClick={() => processVideo(customVideoPath)}
              disabled={isProcessing || !customVideoPath}
              className="w-full bg-gradient-to-r from-green-500 to-green-600 text-white px-4 py-2 rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Processing...' : 'Process Selected Video'}
            </button>
            <button
              onClick={() => previewVideo(customVideoPath)}
              disabled={isPreviewLoading || !customVideoPath}
              className="mt-3 w-full bg-gradient-to-r from-indigo-500 to-indigo-600 text-white px-4 py-2 rounded-lg hover:from-indigo-600 hover:to-indigo-700 transition-all duration-200 shadow-md hover:shadow-lg disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed"
            >
              {isPreviewLoading ? 'Loading Preview...' : 'Preview This Video'}
            </button>
          </div>
        </div>
      </div>
      <div className="mt-8">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Canvas Preview</h3>
        {preview ? (
          <div className="space-y-4">
            <div className="text-sm text-gray-600 space-y-1">
              <p className="font-medium text-gray-700">Currently previewing:</p>
              <p className="truncate">Video: <span className="font-mono text-xs">{preview.absoluteVideoPath}</span></p>
              <p className="truncate">Tracking: <span className="font-mono text-xs">{preview.trackingPath}</span></p>
              <p>Total detections across frames: <span className="font-semibold">{preview.detections}</span></p>
            </div>
            <AnnotatedVideoCanvas
              videoSrc={preview.videoUrl}
              trackingData={preview.trackingData}
            />
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            Load a preview to see the original video rendered on the canvas with bounding box annotations.
          </p>
        )}
      </div>
    </section>
  )
}

export default VideoSection
