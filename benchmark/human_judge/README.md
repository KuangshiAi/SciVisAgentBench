# Human Evaluation UI for SciVisAgentBench

A web-based interface for human judges to evaluate vision-based test cases from SciVisAgentBench. This tool allows evaluators to rate agent-generated visualizations against ground truth images using standardized metrics.

## 🚀 Two Deployment Options

### Option 1: Local Flask Server (Quick Testing)
Run locally for testing or single-user evaluation sessions.

### Option 2: Firebase Deployment (Production)
Deploy online for multi-user collaboration with real-time data sync.
- **See**: [firebase_deploy/README.md](firebase_deploy/README.md) for complete Firebase deployment guide

## 📋 Supported Benchmarks

- `main` - Main visualization benchmark (ParaView tasks)
- `chatvis_bench` - ChatVis benchmark
- `molecular_vis` - Molecular visualization tasks

## 🛠️ Requirements

```bash
pip install flask pyyaml
```

The screenshot generation feature also requires ParaView to be installed and accessible.

## 🎯 Quick Start - Local Flask Server

### Simplified Mode (Recommended - Multiple Benchmarks)

Evaluate cases from multiple benchmarks using a single comprehensive YAML file:

```bash
python -m benchmark.human_judge.run_human_eval \
    --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
    --port 8081
```

This mode:
- ✅ Supports cases from different directories and benchmarks
- ✅ Uses a single YAML file listing all cases to evaluate
- ✅ No need to specify `--agent`, `--config`, `--yaml`, or `--cases`
- ✅ Each case in the YAML includes its path, metrics, and task description

**Example YAML structure** (`selected_15_cases.yaml`):
```yaml
cases:
  - name: argon-bubble
    path: SciVisAgentBench-tasks/main/argon-bubble
    yaml: benchmark/eval_cases/paraview/main_cases.yaml
    description: Color & Opacity Mapping, Volume Rendering

  - name: time-varying
    path: SciVisAgentBench-tasks/chatvis_bench/time-varying
    yaml: benchmark/eval_cases/paraview/chatvis_bench_cases.yaml
    description: Temporal Processing
```

### Legacy Mode (Single Benchmark)

Evaluate a single benchmark with specific agent configuration:

```bash
python -m benchmark.human_judge.run_human_eval \
    --agent chatvis \
    --config benchmark/configs/chatvis/config_openai.json \
    --yaml benchmark/eval_cases/paraview/chatvis_bench_cases.yaml \
    --cases SciVisAgentBench-tasks/chatvis_bench \
    --port 8081
```

Or filter specific cases from the benchmark:

```bash
python -m benchmark.human_judge.run_human_eval \
    --agent paraview_mcp \
    --config benchmark/configs/paraview_mcp/config_openai.json \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --filter-cases benchmark/eval_cases/selected_cases.yaml \
    --port 8081
```

### Access the UI

Once the server starts, open your browser:
```
http://127.0.0.1:8081
```

## 🌐 Firebase Deployment (Production)

For online deployment with multi-user support and real-time data sync:

### Quick Deploy

```bash
# 1. Generate static site
python -m benchmark.human_judge.generate_static_site \
    --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
    --output-dir benchmark/human_judge/firebase_deploy

# 2. Deploy to Firebase
cd benchmark/human_judge/firebase_deploy
firebase deploy
```

**Complete guide**: [firebase_deploy/README.md](firebase_deploy/README.md)

Benefits:
- ☁️ Real-time data sync across multiple evaluators
- 🌍 Accessible worldwide via URL
- 💾 Cloud storage in Firebase Realtime Database
- 📊 Centralized result collection
- 🆓 Free tier supports ~1000 evaluations

## 📝 Command-Line Options

### Simplified Mode (--cases-yaml)

```
--cases-yaml YAML         Path to comprehensive YAML file with case definitions
                          Each case includes: name, path, yaml, description

--output-dir DIR          Directory to save evaluation results
                          (default: benchmark/human_judge/evaluations)

--host HOST               Host to run the server on
                          (default: 127.0.0.1)

--port PORT               Port to run the server on
                          (default: 5000)

--debug                   Run server in debug mode
```

### Legacy Mode (Individual Arguments)

```
--agent AGENT             Name of the agent being evaluated
                          (e.g., chatvis, paraview_mcp, napari_mcp)

--config CONFIG           Path to agent configuration file (JSON)

--yaml YAML               Path to YAML file containing test cases

--cases CASES             Path to directory containing test case data
                          (e.g., SciVisAgentBench-tasks/main)

--filter-cases YAML       Optional: YAML file with list of case names to evaluate
                          Only these cases will be loaded from --yaml

--output-dir DIR          Directory to save evaluation results
                          (default: benchmark/human_judge/evaluations)

--host HOST               Host to run the server on (default: 127.0.0.1)

--port PORT               Port to run the server on (default: 5000)

--debug                   Run server in debug mode
```

## 📊 Evaluation Output

### Local Flask Mode

Evaluations are saved to JSON files in the output directory:

**File naming**: `human_eval_multi_{timestamp}.json` (simplified mode) or `human_eval_{agent}_{benchmark}_{timestamp}.json` (legacy mode)

**Structure**:
```json
{
  "metadata": {
    "mode": "multi_case",
    "cases_yaml": "benchmark/eval_cases/selected_15_cases.yaml",
    "total_cases": 15,
    "unique_yamls": [
      "benchmark/eval_cases/paraview/main_cases.yaml",
      "benchmark/eval_cases/paraview/chatvis_bench_cases.yaml"
    ]
  },
  "timestamp": "20260217_143022",
  "evaluations": [
    {
      "case_index": 0,
      "case_name": "argon-bubble",
      "task_description": "Full task description...",
      "metrics": [
        {"criterion": "Color mapping accurately represents data"},
        {"criterion": "Opacity transfer function allows clear visualization"},
        {"criterion": "Volume rendering quality is high"}
      ],
      "ratings": [8, 7, 9],
      "notes": "Good overall visualization",
      "timestamp": "2026-02-17T14:32:45.123456"
    }
  ]
}
```

### Firebase Mode

Evaluations are saved to Firebase Realtime Database with enhanced metadata:

**Structure**:
```json
{
  "evaluator_name": "John Smith",
  "evaluator_institution": "University of Example",
  "evaluator_email": "john@example.com",
  "case_index": 0,
  "case_name": "argon-bubble",
  "task_description": "Full task description...",
  "metrics": [...],
  "ratings": [8, 7, 9],
  "notes": "Good visualization",
  "timestamp": "2026-02-17T14:32:45.123456",
  "session_id": "20260217143022_abc123xyz"
}
```

**Downloading Firebase results**:
- From Firebase Console: Realtime Database → Export JSON
- From browser: `window.debugHelpers.exportAllEvaluations()`
- Via Python REST API: `requests.get('https://YOUR_PROJECT.firebaseio.com/evaluations.json')`

See [RESULTS_STORAGE.md](eval_cases/RESULTS_STORAGE.md) for detailed format documentation.

## 🎨 Features

### User Interface
- 📱 Responsive design for desktop and mobile
- 🖼️ Side-by-side comparison of ground truth vs agent results
- 📊 Interactive rating sliders (0-10 scale)
- 📝 Optional notes for each evaluation
- ⏭️ Navigation between cases
- 💾 Auto-save functionality (Firebase mode)
- ✅ Real-time save indicator (Firebase mode)

### Supported Media Types
- 🖼️ Images (PNG, JPG)
- 🎬 Videos (MP4, AVI - MP4 recommended for web)
- 📐 Multi-view visualizations (front, side, diagonal)

### Evaluation Metrics
- Automatically extracted from YAML test cases
- Support for numbered criteria formats: `1.` and `1)`
- Custom metrics per case based on visualization type

## 🔧 Advanced Usage

### Generating Static Sites for Deployment

```bash
python -m benchmark.human_judge.generate_static_site \
    --cases-yaml benchmark/eval_cases/selected_15_cases.yaml \
    --output-dir /path/to/output
```

This script:
- Exports case data to JSON
- Copies all images/videos to output directory
- Prepares files for Firebase deployment

### Filtering Cases for Evaluation

Create a filter YAML file listing specific cases:

```yaml
# selected_cases.yaml
cases:
  - argon-bubble
  - bonsai
  - trl-velocity_streamline
```

Then use it:
```bash
python -m benchmark.human_judge.run_human_eval \
    --yaml benchmark/eval_cases/paraview/main_cases.yaml \
    --cases SciVisAgentBench-tasks/main \
    --filter-cases selected_cases.yaml \
    --port 8081
```

## 📚 Additional Documentation

- **Firebase Deployment**: [firebase_deploy/README.md](firebase_deploy/README.md)
- **Quick Start Guide**: [firebase_deploy/QUICK_START.md](firebase_deploy/QUICK_START.md)
- **Static Generation**: [STATIC_GENERATION_GUIDE.md](STATIC_GENERATION_GUIDE.md)
- **Results Format**: [eval_cases/RESULTS_STORAGE.md](eval_cases/RESULTS_STORAGE.md)
- **Full Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🐛 Troubleshooting

### Images not loading
- Check that image paths in YAML are correct
- For 3-view format: Images should be in `evaluation_results/{agent_mode}/screenshots/`
- For single image: Images should be in `results/{agent_mode}/`

### Port already in use
```bash
# Use a different port
python -m benchmark.human_judge.run_human_eval ... --port 8082
```

### Videos not playing
- Use MP4 format (web-compatible)
- AVI files may not play in browsers
- Convert with: `ffmpeg -i video.avi -vcodec h264 video.mp4`

## 💡 Tips

1. **Test locally first**: Use Flask mode to test before deploying to Firebase
2. **Use simplified mode**: When evaluating cases from multiple benchmarks
3. **MP4 for videos**: Always prefer MP4 over AVI for web compatibility
4. **Real-time saves**: Firebase mode auto-saves, so evaluators can pause/resume anytime
5. **Multi-evaluator**: Firebase supports multiple concurrent evaluators

## 🤝 Contributing

When adding new benchmark types:
1. Add test cases to appropriate YAML file in `benchmark/eval_cases/paraview/`
2. Ensure vision-based metrics are defined with `llm-rubric` type
3. Follow numbered format for criteria: `1.` or `1)`
4. Test with both Flask and Firebase modes

## 📄 License

Part of SciVisAgentBench - see main repository for license information.
