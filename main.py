import os
import json
import glob
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Directories
INPUT_JSON_DIR = "input"
PDFS_DIR = "pdfs"
RESULTS_DIR = "output"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load embeddings
model = SentenceTransformer('./models/all-MiniLM-L6-v2') 

# ---------- Embedding ----------
def embed_text(text):
    return model.encode([text], convert_to_numpy=True)[0]


# ---------- Improved Heading Extraction ----------
def extract_headings_and_pages(pdf_path):
    """
    Enhanced heading detection with more flexible criteria
    """
    doc = fitz.open(pdf_path)
    sections = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        # Get all text from page for fallback
        page_text = page.get_text("text").strip()

        # Collect font sizes for this page
        page_font_sizes = []
        for b in blocks:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                for span in l["spans"]:
                    if span["size"] > 0:
                        page_font_sizes.append(span["size"])
        
        if not page_font_sizes:
            continue
            
        median_font = np.median(page_font_sizes)
        max_font = max(page_font_sizes)

        # Extract potential headings
        for b in blocks:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                line_text = " ".join([span["text"].strip() for span in l["spans"]]).strip()
                if not line_text:
                    continue

                # More flexible word count (1-20 words instead of 2-12)
                words = line_text.split()
                if len(words) < 1 or len(words) > 20:
                    continue

                # Less restrictive punctuation filter
                if line_text.count(".") > 3 or line_text.count(",") > 5:
                    continue

                # Get font properties
                font_sizes = [span["size"] for span in l["spans"] if span["size"] > 0]
                if not font_sizes:
                    continue
                    
                avg_font_size = sum(font_sizes) / len(font_sizes)
                is_bold = any(span["flags"] & 2 for span in l["spans"])
                is_upper = line_text.isupper()

                # More lenient heading criteria
                is_heading = (
                    avg_font_size > median_font * 1.1 or  # Slightly larger font
                    is_bold or 
                    is_upper or
                    avg_font_size >= max_font * 0.9  # Close to maximum font size
                )

                if is_heading:
                    sections.append({
                        "section_title": line_text,
                        "page_number": page_num + 1,
                        "pdf_path": pdf_path
                    })

        # Enhanced fallback: extract meaningful first lines and section-like text
        if page_num == 0 or len([s for s in sections if s["page_number"] == page_num + 1]) == 0:
            lines = page_text.split('\n')
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                if len(line) > 10 and len(line.split()) >= 2:
                    # Skip lines that look like body text
                    if not (line.endswith('.') and len(line) > 100):
                        sections.append({
                            "section_title": line,
                            "page_number": page_num + 1,  
                            "pdf_path": pdf_path
                        })
                        break

    # Deduplicate while preserving order
    seen = set()
    unique_sections = []
    for sec in sections:
        # Create a normalized key for comparison
        normalized_title = re.sub(r'\s+', ' ', sec["section_title"].lower().strip())
        key = (normalized_title, sec["page_number"])
        if key not in seen:
            seen.add(key)
            unique_sections.append(sec)

    return unique_sections


# ---------- Enhanced Ranking ----------
def rank_sections(sections, persona, job):
    """
    Improved ranking with better query construction
    """
    # Create a more comprehensive query
    query_text = f"{persona} {job}"
    query_embedding = embed_text(query_text)
    ranked = []

    for sec in sections:
        # Use section title for embedding
        sec_embedding = embed_text(sec["section_title"])
        score = cosine_similarity([query_embedding], [sec_embedding])[0][0]
        ranked.append((sec, score))

    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Get top sections with deduplication
    seen_titles = set()
    top_sections = []
    for sec, score in ranked:
        title_normalized = re.sub(r'\s+', ' ', sec["section_title"].lower().strip())
        if title_normalized not in seen_titles and len(top_sections) < 5:
            seen_titles.add(title_normalized)
            top_sections.append(sec)

    return [
        {
            "document": os.path.basename(sec["pdf_path"]),
            "section_title": sec["section_title"],
            "importance_rank": i + 1,
            "page_number": sec["page_number"]
        }
        for i, sec in enumerate(top_sections)
    ]


# ---------- Enhanced Subsection Extraction ----------
def extract_subsections(ranked_sections, persona, job):
    """
    Better text extraction and processing for subsections
    """
    query_embedding = embed_text(f"{persona} {job}")
    all_paragraphs = []

    for sec in ranked_sections:
        # Find PDF file
        pdf_path = None
        for root, _, files in os.walk(PDFS_DIR):
            if sec["document"] in files:
                pdf_path = os.path.join(root, sec["document"])
                break
        if not pdf_path:
            continue

        try:
            doc = fitz.open(pdf_path)
            page_num = sec["page_number"] - 1
            
            if page_num < 0 or page_num >= len(doc):
                continue

            page = doc.load_page(page_num)
            
            # Get text in multiple ways to capture more content
            text_methods = [
                page.get_text("text"),
                page.get_text("blocks")
            ]
            
            # Process regular text
            page_text = page.get_text("text")
            
            # Better paragraph splitting
            paragraphs = []
            
            # Split by double newlines first
            sections_by_double_newline = page_text.split('\n\n')
            for section in sections_by_double_newline:
                section = section.strip()
                if len(section) > 20:  # Minimum length
                    paragraphs.append(section)
            
            # If no good paragraphs from double newlines, split by single newlines
            if len(paragraphs) < 2:
                lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                current_paragraph = []
                
                for line in lines:
                    if len(line) > 5:  # Skip very short lines
                        current_paragraph.append(line)
                        # End paragraph on sentence endings or if getting long
                        if (line.endswith('.') or line.endswith('!') or line.endswith('?') or 
                            len(' '.join(current_paragraph)) > 150):
                            if len(current_paragraph) > 0:
                                para_text = ' '.join(current_paragraph).strip()
                                if len(para_text) > 30:  # Minimum paragraph length
                                    paragraphs.append(para_text)
                                current_paragraph = []
                
                # Add remaining text as paragraph
                if current_paragraph:
                    para_text = ' '.join(current_paragraph).strip()
                    if len(para_text) > 30:
                        paragraphs.append(para_text)

            # Process each paragraph
            for paragraph in paragraphs:
                if len(paragraph.strip()) < 30:  # Skip very short paragraphs
                    continue
                    
                # Clean up text minimally (don't remove measurements aggressively)
                cleaned_text = re.sub(r'\s+', ' ', paragraph).strip()
                
                # Don't limit to 3 sentences - keep more content
                sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
                
                # Take more sentences but limit total length
                refined_sentences = []
                total_length = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and total_length + len(sentence) < 500:  # Increased limit
                        refined_sentences.append(sentence)
                        total_length += len(sentence)
                    else:
                        break
                
                if refined_sentences:
                    refined_text = ' '.join(refined_sentences).strip()
                    
                    if len(refined_text) > 15:  # Minimum length check
                        # Calculate similarity
                        emb = embed_text(refined_text)
                        score = cosine_similarity([query_embedding], [emb])[0][0]
                        
                        all_paragraphs.append({
                            "document": os.path.basename(pdf_path),
                            "page_number": page_num + 1,
                            "refined_text": refined_text,
                            "similarity_score": float(score)
                        })

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue

    # Sort by similarity and take top 5
    all_paragraphs.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Deduplicate by text similarity
    final_subsections = []
    seen_texts = set()
    
    for para in all_paragraphs:
        # Create a normalized version for comparison
        normalized_text = re.sub(r'[^\w\s]', '', para["refined_text"].lower())
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        # Check if we've seen similar text
        is_duplicate = False
        for seen_text in seen_texts:
            if (len(normalized_text) > 20 and 
                (normalized_text in seen_text or seen_text in normalized_text or
                 len(set(normalized_text.split()) & set(seen_text.split())) > len(normalized_text.split()) * 0.7)):
                is_duplicate = True
                break
        
        if not is_duplicate and len(final_subsections) < 5:
            seen_texts.add(normalized_text)
            # Remove similarity score from final output
            result_para = {k: v for k, v in para.items() if k != "similarity_score"}
            final_subsections.append(result_para)

    return final_subsections


# ---------- Main Processing ----------
def process_round_1b():
    persona_files = glob.glob(os.path.join(INPUT_JSON_DIR, "*.json"))

    for persona_file in persona_files:
        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                persona_data = json.load(f)

            # Extract persona
            persona = persona_data.get("persona", "")
            if isinstance(persona, dict) and "role" in persona:
                persona = persona["role"]

            # Extract job
            job = ""
            if "job" in persona_data:
                job = persona_data["job"]
            elif "job_to_be_done" in persona_data:
                if isinstance(persona_data["job_to_be_done"], dict) and "task" in persona_data["job_to_be_done"]:
                    job = persona_data["job_to_be_done"]["task"]
                else:
                    job = str(persona_data["job_to_be_done"])

            # Get PDF folder
            file_prefix = os.path.splitext(os.path.basename(persona_file))[0]
            pdf_folder = os.path.join(PDFS_DIR, file_prefix)

            if not os.path.exists(pdf_folder):
                print(f"[WARN] PDF folder not found: {pdf_folder}")
                continue

            # Process all PDFs
            all_sections = []
            pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

            if not pdf_files:
                print(f"[WARN] No PDF files found in {pdf_folder}")
                continue

            for pdf_path in pdf_files:
                try:
                    headings = extract_headings_and_pages(pdf_path)
                    all_sections.extend(headings)
                except Exception as e:
                    print(f"[ERROR] Failed to process {pdf_path}: {e}")
                    continue

            if not all_sections:
                print(f"[WARN] No sections extracted for {file_prefix}")
                continue

            # Rank sections and extract subsections
            ranked_sections = rank_sections(all_sections, persona, job)
            subsections = extract_subsections(ranked_sections, persona, job)

            # Create result
            result = {
                "metadata": {
                    "input_documents": [os.path.basename(p) for p in pdf_files],
                    "persona": persona,
                    "job_to_be_done": job,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": ranked_sections,
                "subsection_analysis": subsections
            }

            # Save result
            output_path = os.path.join(RESULTS_DIR, f"{file_prefix}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] Processed {file_prefix} -> {output_path}")
            print(f"  - Extracted {len(ranked_sections)} sections")
            print(f"  - Found {len(subsections)} relevant subsections")

        except Exception as e:
            print(f"[ERROR] Failed to process {persona_file}: {e}")
            continue


if __name__ == "__main__":
    process_round_1b()