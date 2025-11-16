---
name: context7
description: Fetch up-to-date, version-specific library documentation using Context7 MCP. Use when working with third-party libraries (FastAPI, React, pandas, pytest, NumPy, TensorFlow, librosa, soundfile), frameworks, or APIs to get current code examples and avoid outdated patterns. Automatically activates when code involves library-specific imports or when asking about library features.
allowed-tools: Read, Bash
---

# Context7 Documentation Lookup

This skill teaches Claude when and how to use the Context7 MCP server to fetch current library documentation.

## When to Use Context7

Automatically activate this skill when:

1. **Libraries mentioned by name** (FastAPI, React, TensorFlow, pandas, librosa, etc.)
2. **Code involves library-specific APIs** (third-party imports, framework patterns)
3. **Fast-moving ecosystems** (Next.js, React, Tailwind CSS, pytest, transformers)
4. **Version-sensitive scenarios** (breaking changes, deprecated APIs)
5. **Audio/ML libraries** (librosa, soundfile, torch, transformers, MERT)

## When NOT to Use Context7

Skip for:
- Standard library functions (Python built-ins, JavaScript native APIs)
- Stable, well-known patterns (basic NumPy operations, simple pandas queries)
- General programming concepts (loops, conditionals, classes)
- Historical code analysis
- Non-library work (configuration files, documentation writing)

## Workflow

1. **Detect library reference** from user input
2. **Resolve library ID** using `mcp__context7__resolve-library-id` with the library name
3. **Fetch documentation** using `mcp__context7__get-library-docs` with:
   - `context7CompatibleLibraryID`: The ID from step 2 (e.g., `/tiangolo/fastapi`)
   - `topic`: Specific focus area (e.g., "routing", "authentication", "audio loading")
   - `tokens`: Amount of documentation (default: 5000, adjust based on complexity)
4. **Apply documentation** to generate current, correct code

## Token Efficiency Best Practices

- Fetch **one library at a time** with specific queries
- Use the `topic` parameter to focus documentation on relevant features
- Remember library IDs within conversations (don't re-resolve the same library)
- Only fetch when generating library-specific code, not for general discussion
- For well-known stable APIs, skip fetching if confident in the pattern

## Example Invocation Pattern

**User asks:** "Create a FastAPI endpoint with authentication"

**Skill workflow:**
1. Call `mcp__context7__resolve-library-id` with `libraryName: "fastapi"`
   - Returns: `/tiangolo/fastapi`
2. Call `mcp__context7__get-library-docs` with:
   - `context7CompatibleLibraryID: "/tiangolo/fastapi"`
   - `topic: "authentication"`
   - `tokens: 3000`
3. Use the fetched documentation to write current, version-appropriate code

**User asks:** "Load an audio file with librosa"

**Skill workflow:**
1. Call `mcp__context7__resolve-library-id` with `libraryName: "librosa"`
   - Returns: `/librosa/librosa`
2. Call `mcp__context7__get-library-docs` with:
   - `context7CompatibleLibraryID: "/librosa/librosa"`
   - `topic: "loading audio files"`
   - `tokens: 2000`
3. Apply current best practices for audio loading

## Common Libraries for This Project (CCM4)

**Audio Processing:**
- librosa - Audio analysis and feature extraction
- soundfile - Audio file I/O
- scipy.signal - Signal processing

**Machine Learning:**
- transformers - MERT model, Hugging Face models
- torch/tensorflow - Deep learning frameworks
- numpy - Numerical operations

**Data Science:**
- pandas - Data manipulation
- scikit-learn - Machine learning utilities

**Web/API:**
- FastAPI - API endpoints (if adding web interface)
- pydantic - Data validation

**Testing:**
- pytest - Unit testing framework

## Tool Reference

### mcp__context7__resolve-library-id
**Purpose:** Convert library name to Context7-compatible ID

**Parameters:**
- `libraryName` (string, required): Library name to search for (e.g., "fastapi", "librosa")

**Returns:** Library ID in format `/org/project` or `/org/project/version`

**Example:**
```
Input: libraryName = "librosa"
Output: "/librosa/librosa"
```

### mcp__context7__get-library-docs
**Purpose:** Fetch up-to-date documentation and code examples

**Parameters:**
- `context7CompatibleLibraryID` (string, required): ID from resolve-library-id
- `topic` (string, optional): Focus area to narrow documentation (e.g., "routing", "audio loading")
- `tokens` (number, optional, default: 5000): Maximum documentation length

**Returns:** Current documentation with code examples

**Example:**
```
Input:
  context7CompatibleLibraryID = "/librosa/librosa"
  topic = "feature extraction"
  tokens = 3000

Output: Documentation focused on MFCC, chroma, spectral features, etc.
```

## Activation Triggers

This skill should activate automatically when user messages contain:
- Library names: "librosa", "FastAPI", "transformers", "MERT", "torch", "pandas"
- Import statements: "import librosa", "from transformers import"
- Framework mentions: "using FastAPI", "with pytest"
- API-specific terms: "MERT model", "audio loading", "feature extraction"
- Version questions: "latest version", "breaking changes", "deprecated"

## Notes

- This is a **lazy-loading skill** - the context7 MCP server only loads when this skill activates
- Saves ~200-250 tokens in conversations that don't need library documentation
- The MCP server must be configured in `.mcp.json` but won't load until skill invocation
- Prefer this skill over searching documentation manually or using outdated examples
