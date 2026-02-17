// Application state
const appState = {
    evaluator: null,
    evaluatorName: null,
    evaluatorInstitution: null,
    evaluatorEmail: null,
    sessionId: null,
    cases: [],
    currentCaseIndex: 0,
    evaluations: {},
    sessionStarted: false
};

// Load cases from JSON
async function loadCases() {
    try {
        const response = await fetch('data/cases.json');
        const data = await response.json();
        appState.cases = data.cases || [];
        console.log(`Loaded ${appState.cases.length} cases`);
        return true;
    } catch (error) {
        console.error('Error loading cases:', error);
        alert('Failed to load evaluation cases. Please check data/cases.json');
        return false;
    }
}

// Initialize session
async function initSession(evaluatorName, evaluatorInstitution, evaluatorEmail) {
    appState.evaluatorName = evaluatorName;
    appState.evaluatorInstitution = evaluatorInstitution;
    appState.evaluatorEmail = evaluatorEmail;
    appState.evaluator = `${evaluatorName} (${evaluatorInstitution})`;
    appState.sessionId = generateSessionId();
    appState.sessionStarted = true;

    // Create session record in Firebase
    const sessionRef = database.ref(`sessions/${appState.sessionId}`);
    await sessionRef.set({
        evaluator_name: evaluatorName,
        evaluator_institution: evaluatorInstitution,
        evaluator_email: evaluatorEmail,
        started_at: new Date().toISOString(),
        total_cases: appState.cases.length,
        cases_evaluated: 0,
        last_updated: new Date().toISOString()
    });

    // Load previous evaluations for this evaluator
    await loadPreviousEvaluations();

    // Show evaluation panel
    document.getElementById('session-setup').style.display = 'none';
    document.getElementById('evaluation-panel').style.display = 'block';

    // Render first case
    renderCase(0);
    updateProgress();
}

// Generate session ID
function generateSessionId() {
    const timestamp = new Date().toISOString().replace(/[-:T.Z]/g, '').slice(0, 14);
    // Generate random alphanumeric string without periods
    const randomStr = Math.random().toString(36).substring(2).replace(/\./g, '') +
                      Math.random().toString(36).substring(2).replace(/\./g, '');
    return `${timestamp}_${randomStr.substring(0, 9)}`;
}

// Load previous evaluations from Firebase
async function loadPreviousEvaluations() {
    const evaluationsRef = database.ref('evaluations');
    const snapshot = await evaluationsRef.orderByChild('evaluator_email').equalTo(appState.evaluatorEmail).once('value');

    const evaluations = snapshot.val() || {};

    // Map evaluations by case_index
    Object.values(evaluations).forEach(evaluation => {
        appState.evaluations[evaluation.case_index] = evaluation;
    });

    console.log(`Loaded ${Object.keys(appState.evaluations).length} previous evaluations`);
}

// Render current case
function renderCase(index) {
    if (index < 0 || index >= appState.cases.length) {
        return;
    }

    appState.currentCaseIndex = index;
    const caseData = appState.cases[index];

    // Update navigation
    document.getElementById('case-title').textContent = `Case ${index + 1} of ${appState.cases.length}`;
    document.getElementById('case-title-bottom').textContent = `Case ${index + 1} of ${appState.cases.length}`;

    document.getElementById('prev-case').disabled = index === 0;
    document.getElementById('prev-case-bottom').disabled = index === 0;
    document.getElementById('next-case').disabled = index === appState.cases.length - 1;
    document.getElementById('next-case-bottom').disabled = index === appState.cases.length - 1;

    // Update case info
    document.getElementById('case-name').textContent = caseData.name;
    document.getElementById('case-description').textContent = caseData.description || '';
    document.getElementById('task-description').textContent = caseData.task_description || 'No task description available';

    // Render images
    renderImages('ground-truth-images', caseData.ground_truth_images || []);
    renderImages('result-images', caseData.result_images || []);

    // Show/hide no results message
    const noResultsMsg = document.getElementById('no-results-message');
    if (!caseData.result_images || caseData.result_images.length === 0) {
        noResultsMsg.style.display = 'block';
    } else {
        noResultsMsg.style.display = 'none';
    }

    // Render metrics
    renderMetrics(caseData.metrics || []);

    // Load previous evaluation if exists
    const previousEval = appState.evaluations[index];
    if (previousEval) {
        loadPreviousEvaluation(previousEval);
    } else {
        clearEvaluationForm();
    }

    // Scroll to top
    window.scrollTo(0, 0);
}

// Render images/videos in container
function renderImages(containerId, imagePaths) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (!imagePaths || imagePaths.length === 0) {
        container.innerHTML = '<p style="color: #999;">No images available</p>';
        return;
    }

    imagePaths.forEach(imagePath => {
        const isVideo = imagePath.toLowerCase().endsWith('.mp4') ||
                       imagePath.toLowerCase().endsWith('.avi');

        if (isVideo) {
            const mimeType = imagePath.toLowerCase().endsWith('.mp4') ? 'video/mp4' : 'video/x-msvideo';
            const video = document.createElement('video');
            video.controls = true;
            video.style.width = '100%';

            const source = document.createElement('source');
            source.src = getImageUrl(imagePath);
            source.type = mimeType;

            video.appendChild(source);
            container.appendChild(video);
        } else {
            const img = document.createElement('img');
            img.src = getImageUrl(imagePath);
            img.alt = imagePath;
            container.appendChild(img);
        }
    });
}

// Get image URL (Firebase Storage or local)
function getImageUrl(imagePath) {
    // If using Firebase Storage, construct URL
    // For now, use local path (will be replaced with Firebase Storage URLs)
    return `images/${imagePath}`;
}

// Render evaluation metrics
function renderMetrics(metrics) {
    const container = document.getElementById('metrics-list');
    container.innerHTML = '';

    if (!metrics || metrics.length === 0) {
        container.innerHTML = '<p>No evaluation metrics defined for this case</p>';
        return;
    }

    metrics.forEach((metric, index) => {
        const metricDiv = document.createElement('div');
        metricDiv.className = 'metric-item';

        const label = document.createElement('div');
        label.className = 'metric-label';
        label.textContent = `${index + 1}. ${metric.criterion}`;

        const controls = document.createElement('div');
        controls.className = 'rating-controls';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'rating-slider';
        slider.min = 0;
        slider.max = 10;
        slider.value = 5;
        slider.id = `metric-${index}`;

        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'rating-value';
        valueDisplay.textContent = '5 / 10';
        valueDisplay.id = `value-${index}`;

        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = `${e.target.value} / 10`;
        });

        controls.appendChild(slider);
        controls.appendChild(valueDisplay);

        metricDiv.appendChild(label);
        metricDiv.appendChild(controls);

        container.appendChild(metricDiv);
    });
}

// Load previous evaluation into form
function loadPreviousEvaluation(evaluation) {
    // Load ratings
    if (evaluation.ratings) {
        evaluation.ratings.forEach((rating, index) => {
            const slider = document.getElementById(`metric-${index}`);
            const valueDisplay = document.getElementById(`value-${index}`);
            if (slider) {
                slider.value = rating;
                valueDisplay.textContent = `${rating} / 10`;
            }
        });
    }

    // Load notes
    const notesField = document.getElementById('notes');
    notesField.value = evaluation.notes || '';

    // Show success message
    showSuccessMessage('Previous evaluation loaded');
}

// Clear evaluation form
function clearEvaluationForm() {
    // Reset all sliders to 5
    const sliders = document.querySelectorAll('.rating-slider');
    sliders.forEach((slider, index) => {
        slider.value = 5;
        document.getElementById(`value-${index}`).textContent = '5 / 10';
    });

    // Clear notes
    document.getElementById('notes').value = '';
}

// Show success message
function showSuccessMessage(message) {
    const existingMessage = document.querySelector('.success-message');
    if (existingMessage) {
        existingMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'success-message';
    messageDiv.textContent = `✓ ${message}`;

    const form = document.querySelector('.evaluation-form');
    form.insertBefore(messageDiv, form.firstChild);

    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

// Update progress bar
function updateProgress() {
    const evaluatedCount = Object.keys(appState.evaluations).length;
    const totalCases = appState.cases.length;
    const percentage = totalCases > 0 ? (evaluatedCount / totalCases) * 100 : 0;

    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${evaluatedCount} / ${totalCases} cases evaluated`;

    // Update session in Firebase
    if (appState.sessionId) {
        database.ref(`sessions/${appState.sessionId}`).update({
            cases_evaluated: evaluatedCount,
            last_updated: new Date().toISOString()
        });
    }
}

// Navigate to case
function navigateToCase(direction) {
    const newIndex = appState.currentCaseIndex + direction;
    if (newIndex >= 0 && newIndex < appState.cases.length) {
        renderCase(newIndex);
    }
}

// Export results to JSON
function exportResults() {
    const data = {
        evaluator_name: appState.evaluatorName,
        evaluator_institution: appState.evaluatorInstitution,
        evaluator_email: appState.evaluatorEmail,
        session_id: appState.sessionId,
        timestamp: new Date().toISOString(),
        evaluations: Object.values(appState.evaluations)
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const filename = `evaluations_${appState.evaluatorName.replace(/\s+/g, '_')}_${appState.sessionId}.json`;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Event listeners
document.addEventListener('DOMContentLoaded', async () => {
    // Load cases
    const success = await loadCases();
    if (!success) return;

    // Start session button
    document.getElementById('start-session').addEventListener('click', () => {
        const evaluatorName = document.getElementById('evaluator-name').value.trim();
        const evaluatorInstitution = document.getElementById('evaluator-institution').value.trim();
        const evaluatorEmail = document.getElementById('evaluator-email').value.trim();

        // Validate all fields
        if (!evaluatorName) {
            alert('Please enter your full name');
            return;
        }
        if (!evaluatorInstitution) {
            alert('Please enter your institution');
            return;
        }
        if (!evaluatorEmail) {
            alert('Please enter your email');
            return;
        }

        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(evaluatorEmail)) {
            alert('Please enter a valid email address');
            return;
        }

        initSession(evaluatorName, evaluatorInstitution, evaluatorEmail);
    });

    // Navigation buttons
    document.getElementById('prev-case').addEventListener('click', () => navigateToCase(-1));
    document.getElementById('next-case').addEventListener('click', () => navigateToCase(1));
    document.getElementById('prev-case-bottom').addEventListener('click', () => navigateToCase(-1));
    document.getElementById('next-case-bottom').addEventListener('click', () => navigateToCase(1));

    // Save evaluation button
    document.getElementById('save-evaluation').addEventListener('click', saveEvaluation);

    // Skip case button
    document.getElementById('skip-case').addEventListener('click', () => {
        if (appState.currentCaseIndex < appState.cases.length - 1) {
            navigateToCase(1);
        }
    });

    // Export results
    document.getElementById('export-results').addEventListener('click', exportResults);

    // Allow Enter key to start session
    document.getElementById('evaluator-name').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('start-session').click();
        }
    });
});
