# PDF Outline Extraction – Round 1B

This project extracts structured outlines (headings + subsections) from multiple PDFs, ranks them by importance using persona/job prompts, and generates JSON output.

---

## **How It Works**

### **1. Input**
- JSON in `input/` folder:
  ```json
  {
    "persona": {"role": "Travel Planner"},
    "job_to_be_done": {"task": "Plan a trip of 4 days for a group of 10 college friends."},
    "documents": [
      {"filename": "South of France - Cities.pdf", "title": "South of France - Cities"},
      ...
    ]
  }
