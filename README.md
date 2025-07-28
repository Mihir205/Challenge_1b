# PDF Outline & Subsection Extractor – Adobe Hackathon Round 1B

This project extracts **structured outlines (headings) and relevant subsections** from multiple PDFs, guided by **persona and job-to-be-done** context provided in an input JSON. The system produces a clean **JSON output** ranking important sections and summarizing key content.

---

## **Features**

- **Font & Layout Heuristics**
  - Detects headings using **font size, bold, capitalization, and layout position**.
  - Filters out **noise** like footers, headers, page numbers, and repeated lines.
  - Supports **multi-line heading detection** and merging.

- **Semantic Ranking**
  - Embeds headings and persona/job context using `sentence-transformers`.
  - Ranks top **5 most relevant headings** via cosine similarity.

- **Contextual Subsection Analysis**
  - Extracts related paragraphs from ranked pages.
  - Produces **humanized summaries** aligned to persona/job goals.
  - Removes redundant or overlapping text intelligently.

- **Robust Multi-Document Handling**
  - Processes entire PDF collections (challenge inputs).
  - Works across **varied domains** (travel, food, technical, etc.).

- **Structured JSON Output**
  - Includes metadata, ranked sections, and summarized subsections:
  ```json
  {
    "metadata": {
      "input_documents": ["file1.pdf", "file2.pdf"],
      "persona": "Travel Planner",
      "job_to_be_done": "Plan a 4-day college trip",
      "processing_timestamp": "2025-07-26T15:14:06"
    },
    "extracted_sections": [
      { "document": "file1.pdf", "section_title": "Introduction", "importance_rank": 1, "page_number": 1 }
    ],
    "subsection_analysis": [
      { "document": "file1.pdf", "page_number": 1, "refined_text": "Concise humanized summary..." }
    ]
  }
