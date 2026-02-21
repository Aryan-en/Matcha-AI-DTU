# 🤝 Contributing to Matcha-AI-DTU

First off, thank you for considering contributing to Matcha-AI-DTU! This project is a complex monorepo blending Next.js, NestJS, and Python CV/LLM pipelines. We're excited to have you on board.

## 1. Where to Start?
If you are looking for a place to contribute:
1. Check out the **Issues** tab.
2. Look for tickets tagged `good first issue` or `help wanted`.
3. Read the `docs/ARCHITECTURE.md` to understand the overarching data flow before jumping in.

## 2. Fork & Clone
1. Fork the repository on GitHub.
2. Clone your forked repo to your local machine.
3. Add the upstream remote: `git remote add upstream [repository url]`

## 3. Branch Naming Convention
We strictly follow a structured branch naming paradigm to keep CI/CD pipelines happy:

* `feature/issue-number-short-description` (e.g. `feature/42-add-tts-voices`)
* `bugfix/issue-number-short-description` (e.g. `bugfix/99-fix-websocket-crash`)
* `docs/short-description`
* `refactor/component-name`

## 4. Commit Message Standard
We utilize **Conventional Commits**:
- `feat: [description]` for new features.
- `fix: [description]` for bug fixes.
- `docs: [description]` for documentation alterations.
- `style: [description]` styling/formatting (prettier, eslint changes).
- `refactor: [description]` refactoring existing logic without breaking API boundaries.
- `chore: [description]` updating dependencies or CI pipelines.

*Example*: `feat: integrate YOLOv8n object detection model for goal calibration`

## 5. Development Workflow Recommendations
- **Always run linters** before opening a PR. Ensure `npm run lint` and `npm run format` pass successfully in the root directory.
- For Python code in `services/inference`, please use appropriate type hinting for FastAPI Pydantic models. We try to keep Python styling PEP 8 compliant.

## 6. Pull Requests
1. All Development takes place on the `dev` branch. Pull requests should target `dev`, not `main`.
2. When creating a PR, the `.github/PULL_REQUEST_TEMPLATE.md` will automatically populate. **Please fill it out fully.**
3. Reference the Issue number utilizing closing terminology (e.g. "Closes #42").
4. Wait for Code Reviews. At least 1 approval is needed before merging.

Thank you! Let's build the best sports ML platform.
