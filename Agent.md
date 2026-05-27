# agents.md

# Multi-Modal Learning RAG Agent System

## Overview

This project uses a multi-agent architecture to process and understand:

- YouTube lectures
- PDFs
- Study notes
- Transcripts

The agents collaborate to generate:

- summaries
- quizzes
- flashcards
- timestamp references
- personalized study plans

---

# Agent Architecture

## 1. Retrieval Agent

### Responsibilities

- Retrieve relevant chunks from vector database
- Perform semantic search
- Combine transcript + PDF context
- Return top-k relevant passages

### Input

- User query

### Output

- Ranked relevant chunks

### Tools

- FAISS
- Embedding model
- Hybrid retrieval

---

## 2. Notes Agent

### Responsibilities

- Generate concise study notes
- Create topic-wise summaries
- Extract important definitions
- Highlight formulas and key points

### Input

- Retrieved educational context

### Output

- Structured notes

---

## 3. Quiz Agent

### Responsibilities

- Generate MCQs
- Generate short-answer questions
- Create difficulty-based quizzes

### Modes

- Beginner
- Intermediate
- Exam Mode

---

## 4. Flashcard Agent

### Responsibilities

- Generate flashcards
- Extract concepts and definitions
- Create revision material

### Output Format

Front:
Back:

---

## 5. Timestamp Agent

### Responsibilities

- Identify lecture timestamps
- Map concepts to video positions
- Return exact explanation times

### Example

"Backpropagation explained at 12:45"

---

## 6. Citation Agent

### Responsibilities

- Verify retrieved context
- Attach source references
- Reduce hallucinations

### Output

- Source file
- Page number
- Timestamp

---

## 7. Planner Agent

### Responsibilities

- Generate personalized study plans
- Estimate learning time
- Organize topics by difficulty

### Example

"Complete DBMS in 7 days"

---

# Workflow

User Query
↓
Retrieval Agent
↓
Context Validation
↓
Task-specific Agent
↓
Citation Agent
↓
Final Response

---

# Future Improvements

- Voice interaction
- OCR support
- Handwritten note understanding
- Multi-language support
- Long-term memory
- Knowledge graph generation
