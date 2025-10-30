# Developer Rules for Code Editing and Debugging (VS Code Edition)

<!--  
  This document defines core principles and quality standards  
  for all development done in VS Code.  
  Follow these consistently to ensure clarity, precision, and long-term maintainability.
-->

---

## 1. Context Before Code
- **Understand the context first** — read related files, docs, and code comments.  
- Inspect the project structure and dependencies before making changes.  
- Use VS Code’s built-in search and outline tools to trace logic paths.  
- Never guess what needs changing — **confirm** through inspection and reasoning.

---

## 2. Make Focused, Minimal Edits
- Keep each edit **small, specific, and purposeful**.  
- Avoid large refactors unless they’re absolutely necessary and well-scoped.  
- Prefer surgical edits that improve clarity or correctness.  
- Each commit should have a **single clear intent**.

---

## 3. Iterative Debugging
- Start from the simplest working state.  
- Use breakpoints, logging (`console.log`, `print`, etc.), and VS Code’s debugger effectively.  
- Validate one step at a time — commit progress incrementally.  
- Apply the **“while not working, diagnose and fix”** principle methodically.

---

## Quality Control Standards

### 4. Avoid Heuristics and Hacks
- Don’t rely on “reasonable guesses” when specs or documentation are explicit.  
- If a heuristic feels necessary, step back and analyze the real issue.  
- Fix root causes, not surface symptoms.  
- Trust formal definitions and specifications over intuition.

---

### 5. Maintain Structural Integrity
- Never remove tests or disable checks just to make things compile or pass.  
- Don’t sacrifice readability or design for quick fixes.  
- Revert or rewrite changes that don’t clearly improve the codebase.  
- Keep code clean, consistent, and logically structured.

---

### 6. Manage Resources Precisely
- Allocate and release resources deterministically.  
- Avoid arbitrary buffer sizes or magic numbers.  
- Be explicit about performance and memory requirements.  
- Ensure predictable behavior under both load and idle conditions.

---

## Problem-Solving Discipline

### 7. Systematic Debugging
- Use VS Code’s debugger, logging, and watch variables extensively.  
- Break complex issues into smaller, reproducible test cases.  
- Use binary inspectors or hexdump tools when debugging low-level formats.  
- Don’t rely solely on visual inspection of output.

---

### 8. Think System-Wide
- Understand how your changes affect the **entire system**.  
- Use refactors to reduce technical debt and improve modularity.  
- Employ intermediate representations when they clarify logic or data flow.  
- Follow idiomatic language patterns and conventions.

---

### 9. Focus and Persistence
- Work in **focused sessions** (20–40 minutes of undistracted problem-solving).  
- Don’t abandon an approach after a single failure — iterate with insight.  
- Keep a clear mental model of the end goal.  
- Focus on correctness and clarity over quick success.

---

## Communication & Collaboration

### 10. Clear Task Definition
- Write or request **specific, actionable** tasks.  
- Avoid vague requests like “make it better.”  
- Include concrete examples or reference points when giving feedback.  
- Define clear success criteria for each change or issue.

---

### 11. Guided Autonomy
- Allow exploration and reasoning in implementation.  
- Encourage iteration and backtracking when needed.  
- Provide **direction**, not micromanagement.  
- Trust developers (or automated systems) to make strong architectural choices.

---

## Implementation Practices

### 12. Testing and Verification
- Test components individually before integration.  
- Use unit tests, snapshots, or comparisons to reference outputs when available.  
- Follow deterministic specs and avoid “mostly working” results.  
- Commit only when functionality is **verified and reproducible**.

---

### 13. Documentation and Logging
- Add logging and comments early in development.  
- Document any complex logic or system-specific behavior.  
- Keep inline notes for binary handling, formats, or protocol details.  
- Ensure debug logs are **useful, readable, and toggleable** (via flags or levels).

---

<!-- End of Developer Rules for VS Code -->
