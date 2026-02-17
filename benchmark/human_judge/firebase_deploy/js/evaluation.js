// Save evaluation to Firebase
async function saveEvaluation() {
    const caseData = appState.cases[appState.currentCaseIndex];

    // Collect ratings
    const ratings = [];
    const metrics = caseData.metrics || [];

    for (let i = 0; i < metrics.length; i++) {
        const slider = document.getElementById(`metric-${i}`);
        if (slider) {
            ratings.push(parseInt(slider.value));
        }
    }

    // Get notes
    const notes = document.getElementById('notes').value.trim();

    // Validate
    if (ratings.length === 0) {
        alert('Please rate at least one metric');
        return;
    }

    // Create evaluation object
    const evaluation = {
        evaluator_name: appState.evaluatorName,
        evaluator_institution: appState.evaluatorInstitution,
        evaluator_email: appState.evaluatorEmail,
        case_index: appState.currentCaseIndex,
        case_name: caseData.name,
        task_description: caseData.task_description || '',
        metrics: metrics,
        ratings: ratings,
        notes: notes,
        timestamp: new Date().toISOString(),
        session_id: appState.sessionId
    };

    try {
        // Save to Firebase
        const evaluationRef = database.ref('evaluations').push();
        await evaluationRef.set(evaluation);

        // Update local state
        appState.evaluations[appState.currentCaseIndex] = evaluation;

        // Show success message
        showSuccessMessage('Evaluation saved successfully!');

        // Show real-time save indicator
        showSaveIndicator();

        // Update progress
        updateProgress();

        // Auto-advance to next case after 1 second
        setTimeout(() => {
            if (appState.currentCaseIndex < appState.cases.length - 1) {
                navigateToCase(1);
            } else {
                showCompletionMessage();
            }
        }, 1000);

    } catch (error) {
        console.error('Error saving evaluation:', error);
        alert('Failed to save evaluation. Please try again.');
    }
}

// Show completion message
function showCompletionMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'success-message';
    messageDiv.innerHTML = `
        <h3>🎉 All cases evaluated!</h3>
        <p>Thank you for completing the evaluation.</p>
        <p>Your results have been saved to Firebase.</p>
        <button onclick="exportResults()" style="margin-top: 10px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">
            📥 Export Results
        </button>
    `;

    const form = document.querySelector('.evaluation-form');
    form.innerHTML = '';
    form.appendChild(messageDiv);
}

// Real-time progress sync (optional)
function setupRealtimeSync() {
    if (!appState.sessionId) return;

    // Listen to evaluations for this session
    const evaluationsRef = database.ref('evaluations');
    evaluationsRef.orderByChild('session_id').equalTo(appState.sessionId).on('child_added', (snapshot) => {
        const evaluation = snapshot.val();
        if (evaluation.evaluator === appState.evaluator) {
            appState.evaluations[evaluation.case_index] = evaluation;
            updateProgress();
        }
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (!appState.sessionStarted) return;

    // Arrow keys for navigation
    if (e.key === 'ArrowLeft' && appState.currentCaseIndex > 0) {
        navigateToCase(-1);
    } else if (e.key === 'ArrowRight' && appState.currentCaseIndex < appState.cases.length - 1) {
        navigateToCase(1);
    }

    // Ctrl+S or Cmd+S to save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveEvaluation();
    }
});

// Helper: Get all evaluations from Firebase (for admin)
async function getAllEvaluations() {
    const snapshot = await database.ref('evaluations').once('value');
    return snapshot.val() || {};
}

// Helper: Get evaluations by evaluator
async function getEvaluationsByEvaluator(evaluatorName) {
    const snapshot = await database.ref('evaluations')
        .orderByChild('evaluator')
        .equalTo(evaluatorName)
        .once('value');
    return snapshot.val() || {};
}

// Helper: Export all evaluations (for admin)
async function exportAllEvaluations() {
    const evaluations = await getAllEvaluations();

    const blob = new Blob([JSON.stringify(evaluations, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `all_evaluations_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Make helper functions available globally for debugging
window.debugHelpers = {
    getAllEvaluations,
    getEvaluationsByEvaluator,
    exportAllEvaluations
};

// Show real-time save indicator
function showSaveIndicator() {
    const indicator = document.getElementById('save-indicator');
    indicator.classList.add('show');

    // Hide after 3 seconds
    setTimeout(() => {
        indicator.classList.remove('show');
    }, 3000);
}
