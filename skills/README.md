# SciVisAgentBench Skills

This directory contains **Claude Code skill files** — embedded expertise that enables Claude to perform specialized scientific visualization tasks headlessly and automatically.

Skills are defined as Markdown files deployed to `~/.claude/commands/`. Once deployed, they can be invoked as slash commands (e.g. `/napari-viz`) or triggered automatically via `CLAUDE.md` instructions.

---

## Directory Structure

```
skills/
├── README.md                  # This file
├── napari-viz.md              # Napari headless visualization
├── vmd-mdanalysis-viz.md      # VMD molecular visualization + MDAnalysis trajectory analysis
├── matplotlib-viz.md          # (planned) Matplotlib scientific plotting
└── ...
```

---

## Setup

### 1. Install a skill globally

Copy (or symlink) a skill file to `~/.claude/commands/`:

```bash
# Copy
cp skills/napari-viz.md ~/.claude/commands/napari-viz.md

# Or symlink (stays in sync with repo changes)
ln -sf "$(pwd)/skills/napari-viz.md" ~/.claude/commands/napari-viz.md
```

### 2. Enable auto-triggering via CLAUDE.md

Add a trigger line to the project `CLAUDE.md` (already done for napari):

```
For any napari or scientific image visualization task, automatically invoke /napari-viz with the task description.
```

The `CLAUDE.md` at the repo root is loaded automatically whenever Claude Code runs inside this project.

### 3. Verify the skill is available

Open a Claude Code session and run `/napari-viz Create a random 100x100 image and save a screenshot`. You should get a rendered PNG without any GUI appearing.

---

## Adding a New Skill

1. Create `skills/<tool-name>-viz.md` following the structure of `napari-viz.md`:
   - **Header & role definition** — what the tool does, how Claude should behave
   - **Rules** — environment setup, headless flags, required packages
   - **Canonical script template** — copy-pasteable boilerplate
   - **API reference** — key commands translated to scripting API
   - **Workflow patterns** — common use cases
   - **Debugging & error handling** — known failure modes and fixes
   - **Task execution** — how to process `$ARGUMENTS`

2. Deploy it: `cp skills/<tool-name>-viz.md ~/.claude/commands/`

3. Add a trigger to `CLAUDE.md`:
   ```
   For any <tool> visualization task, automatically invoke /<tool-name>-viz with the task description.
   ```

---

## Available Skills

| Skill file | Command | Trigger condition |
|------------|---------|-------------------|
| `napari-viz.md` | `/napari-viz` | napari or scientific image visualization tasks |
| `vmd-mdanalysis-viz.md` | `/vmd-mdanalysis-viz` | VMD molecular visualization, MDAnalysis trajectory analysis, or molecular dynamics tasks |

---

## Portability

Skills are plain Markdown — no server, no daemon, no MCP. They work on any machine with Claude Code installed. The only requirement is that the underlying tool (e.g. napari) is installed in the active Python environment.
