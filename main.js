// Import Tauri API
const { invoke } = window.__TAURI__.core;

// DOM Elements
const systemInfoBtn = document.getElementById('refresh-system-info');
const systemInfoDisplay = document.getElementById('system-info-display');
const yoloTrackerBtn = document.getElementById('start-yolo-tracker');
const transformerTrackerBtn = document.getElementById('start-transformer-tracker');
const menuTrackerBtn = document.getElementById('start-menu-tracker');
const refreshVideosBtn = document.getElementById('refresh-videos');
const videoSelect = document.getElementById('video-select');
const processVideoBtn = document.getElementById('process-video');
const customVideoPath = document.getElementById('custom-video-path');
const processCustomVideoBtn = document.getElementById('process-custom-video');
const clearOutputBtn = document.getElementById('clear-output');
const outputDisplay = document.getElementById('output-display');

// Utility Functions
function addToOutput(message, isError = false) {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = isError ? '[ERROR]' : '[INFO]';
    const formattedMessage = `${timestamp} ${prefix} ${message}\n`;

    outputDisplay.textContent += formattedMessage;
    outputDisplay.scrollTop = outputDisplay.scrollHeight;
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = 'Loading...';
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText;
        button.classList.remove('loading');
    }
}

function showError(message) {
    addToOutput(message, true);
    alert(`Error: ${message}`);
}

// System Info Functions
async function refreshSystemInfo() {
    setButtonLoading(systemInfoBtn, true);

    try {
        const info = await invoke('get_system_info');
        systemInfoDisplay.textContent = info;
        addToOutput('System information refreshed successfully');
    } catch (error) {
        systemInfoDisplay.textContent = `Error: ${error}`;
        showError(`Failed to get system info: ${error}`);
    } finally {
        setButtonLoading(systemInfoBtn, false);
    }
}

// Tracker Functions
async function startTracker(trackerType, buttonElement) {
    setButtonLoading(buttonElement, true);
    addToOutput(`Starting ${trackerType} tracker...`);

    try {
        const result = await invoke('run_python_tracker', { trackerType });
        addToOutput(`${trackerType} tracker output:\n${result}`);
    } catch (error) {
        showError(`Failed to start ${trackerType} tracker: ${error}`);
    } finally {
        setButtonLoading(buttonElement, false);
    }
}

// Video Functions
async function refreshVideoList() {
    setButtonLoading(refreshVideosBtn, true);

    try {
        const videoFiles = await invoke('list_video_files');

        // Clear existing options except the first one
        videoSelect.innerHTML = '<option value="">Select a video file...</option>';

        if (videoFiles.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No video files found in data/ directory';
            option.disabled = true;
            videoSelect.appendChild(option);
            addToOutput('No video files found in data/ directory');
        } else {
            videoFiles.forEach(filename => {
                const option = document.createElement('option');
                option.value = `data/${filename}`;
                option.textContent = filename;
                videoSelect.appendChild(option);
            });
            addToOutput(`Found ${videoFiles.length} video file(s)`);
        }
    } catch (error) {
        showError(`Failed to refresh video list: ${error}`);
    } finally {
        setButtonLoading(refreshVideosBtn, false);
    }
}

async function processVideo(videoPath, buttonElement) {
    if (!videoPath || videoPath.trim() === '') {
        showError('Please select or specify a video file');
        return;
    }

    setButtonLoading(buttonElement, true);
    addToOutput(`Processing video: ${videoPath}`);

    try {
        const result = await invoke('process_video', { videoPath });
        addToOutput(`Video processing completed:\n${result}`);
        addToOutput(`Output video should be saved as: ${videoPath.replace(/\.[^/.]+$/, '_output.mp4')}`);
    } catch (error) {
        showError(`Failed to process video: ${error}`);
    } finally {
        setButtonLoading(buttonElement, false);
    }
}

// Event Listeners
systemInfoBtn.addEventListener('click', refreshSystemInfo);

yoloTrackerBtn.addEventListener('click', () => {
    startTracker('yolo', yoloTrackerBtn);
});

transformerTrackerBtn.addEventListener('click', () => {
    startTracker('transformer', transformerTrackerBtn);
});

menuTrackerBtn.addEventListener('click', () => {
    startTracker('run_menu', menuTrackerBtn);
});

refreshVideosBtn.addEventListener('click', refreshVideoList);

videoSelect.addEventListener('change', () => {
    processVideoBtn.disabled = !videoSelect.value;
});

processVideoBtn.addEventListener('click', () => {
    processVideo(videoSelect.value, processVideoBtn);
});

processCustomVideoBtn.addEventListener('click', () => {
    processVideo(customVideoPath.value, processCustomVideoBtn);
});

clearOutputBtn.addEventListener('click', () => {
    outputDisplay.textContent = 'Output cleared...\n';
});

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    addToOutput('EDTH Object Tracker GUI initialized');
    addToOutput('Click "Refresh System Info" to check Python environment');
    addToOutput('Click "Refresh Video List" to see available videos in data/ directory');

    // Auto-refresh video list on startup
    refreshVideoList();
});