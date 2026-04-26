from __future__ import annotations

VLM_OCR_RAW = (
    "Transcribe all text visible on this newspaper page exactly as printed. "
    "Preserve the original Italian text, line breaks, and formatting. "
    "Include headlines, articles, advertisements, and notices. "
    "Output only the transcribed text, nothing else."
)


VLM_OCR_MARKDOWN_NATIVE = (
    "Return the text content of this document in Markdown format. "
    "Use # for main titles/headlines, ## for article headlines, and plain text for body paragraphs. "
    "Separate distinct articles with two blank lines. "
    "Preserve the original Italian text exactly as printed."
)

VLM_OCR_STRUCTURED = (
    "Transcribe this newspaper page and identify distinct content units "
    "(articles, advertisements, obituaries, notices). For each unit, provide:\n"
    '- "unit_type": the type of content\n'
    '- "headline": the headline if present, or null\n'
    '- "text": the full transcribed text\n'
    '- "page_span": [page_number]\n\n'
    'Output valid JSON with a single key "articles" containing a list of these units.\n'
    "Preserve the original Italian text exactly as printed."
)

LLM_CLEANUP = (
    "You are given raw OCR output from a historical Italian newspaper issue. "
    "The text comes from {page_count} pages.\n\n"
    "Your task:\n"
    "1. Correct obvious OCR errors while preserving the original Italian\n"
    "2. Identify and separate distinct content units (articles, ads, obituaries, notices)\n"
    "3. Determine reading order across pages (articles may span multiple pages)\n"
    "4. For each unit, determine which pages it appears on\n\n"
    "Raw OCR text:\n{text}\n\n"
    'Output valid JSON with a single key "articles" containing a list of objects, each with:\n'
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial" | "other"\n'
    '- "headline": headline text or null\n'
    '- "paragraphs": list of {{"text": "paragraph text"}}\n'
    '- "page_span": list of page numbers where this unit appears\n\n'
    "Order the articles by their position in the newspaper (front page first, left to right, top to bottom)."
)

VLM_OCR_RAW_V2 = (
    "You are an expert OCR system for historical Italian newspapers from 1880-1945. "
    "This page has multiple columns of dense text.\n\n"
    "CRITICAL: Read each column top-to-bottom, then move to the next column left-to-right. "
    "Do NOT read across columns.\n\n"
    "Transcribe ALL text on this page exactly as printed, preserving the original Italian. "
    "Include every headline, article, advertisement, obituary, and notice. "
    "Mark column breaks with [COL] and article breaks with [ART].\n"
    "Output ONLY the transcribed text."
)

VLM_OCR_STRUCTURED_V2 = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "This page has multiple columns of dense Italian text in old typefaces.\n\n"
    "CRITICAL RULES:\n"
    "1. Read each column TOP-TO-BOTTOM before moving to the next column LEFT-TO-RIGHT\n"
    "2. Transcribe ALL text — do not skip or summarize anything\n"
    "3. Preserve the original Italian exactly as printed, including archaic spelling\n"
    "4. Separate distinct content units (articles, ads, obituaries, notices)\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial"\n'
    '- "headline": the headline text or null\n'
    '- "text": the COMPLETE transcribed text (do not truncate)\n'
    '- "page_span": [page_number]\n\n'
    'Output valid JSON: {"articles": [...]}\n'
    "Do NOT wrap in markdown code blocks. Output raw JSON only."
)

VLM_OCR_STRUCTURED_V2_NOTHINK = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "This image shows columns of dense Italian text in old typefaces.\n\n"
    "CRITICAL RULES:\n"
    "1. Read each column TOP-TO-BOTTOM before moving to the next column LEFT-TO-RIGHT\n"
    "2. Transcribe ALL text — do not skip or summarize anything\n"
    "3. Preserve the original Italian exactly as printed, including archaic spelling\n"
    "4. Separate distinct content units (articles, ads, obituaries, notices)\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial"\n'
    '- "headline": the headline text or null\n'
    '- "text": the COMPLETE transcribed text (do not truncate)\n'
    '- "page_span": [page_number]\n\n'
    'Output valid JSON: {"articles": [...]}\n'
    "Do NOT wrap in markdown code blocks. Output raw JSON only. /no_think"
)

VLM_OCR_COLUMN = (
    "Transcribe ALL text in this newspaper column exactly as printed. "
    "This is a single column cropped from a historical Italian newspaper page (1880-1945). "
    "Preserve the original Italian text, including archaic spelling. "
    "Include every word — do not skip or summarize. "
    "Output only the transcribed text, nothing else. /no_think"
)

LLM_CLEANUP_V2 = (
    "You are given raw OCR output from a historical Italian newspaper issue ({page_count} pages). "
    "The text was extracted column-by-column from left to right, top to bottom.\n\n"
    "Your tasks:\n"
    "1. Fix obvious OCR errors (rn→m, l→I, ſ→s, broken words) while preserving original Italian\n"
    "2. Identify distinct content units: articles, advertisements, obituaries, notices, editorials\n"
    "3. Determine reading order and which pages each unit spans\n"
    "4. Split articles into paragraphs\n\n"
    "Raw OCR text:\n{text}\n\n"
    'Output valid JSON (no markdown code blocks): {{"articles": [...]}}\n'
    "Each article: "
    '{{"unit_type": "...", "headline": "..." or null, "paragraphs": [{{"text": "..."}}], "page_span": [1, 2]}}\n'
    "Order by position in newspaper. Include ALL text — do not truncate or summarize."
)

LLM_POST_CORRECTION = (
    "You are a text correction expert for historical Italian newspapers (1880-1945). "
    "Below is OCR output that may contain errors.\n\n"
    "Fix ONLY obvious OCR errors:\n"
    "- Character confusions: rn→m, cl→d, li→h, 0→o, l→i, etc.\n"
    "- Broken words: rejoin words split across lines\n"
    "- Missing/extra spaces\n"
    "- Garbled sequences from column merging\n\n"
    "Do NOT:\n"
    "- Modernize archaic Italian spelling\n"
    "- Change the meaning or content\n"
    "- Add or remove text\n"
    "- Translate anything\n\n"
    "Text to correct:\n{text}\n\n"
    "Output the corrected text only."
)

VLM_OCR_STRUCTURED_V3 = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "This image shows columns of dense Italian text in old typefaces.\n\n"
    "CRITICAL RULES:\n"
    "1. Read each column TOP-TO-BOTTOM before moving to the next column LEFT-TO-RIGHT\n"
    "2. Transcribe ALL text — do not skip or summarize anything\n"
    "3. Preserve the original Italian exactly as printed, including archaic spelling\n"
    "4. Separate EVERY distinct content unit: news articles, editorials, advertisements, "
    "classified ads, legal notices, obituaries, event listings, serialized fiction, "
    "financial reports, weather, and any other distinct section\n"
    "5. Even small ads and brief notices are separate content units — do not skip them\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial" | "fiction" | "classified"\n'
    '- "headline": the headline or title text, or null if none\n'
    '- "text": the COMPLETE transcribed text (do not truncate)\n'
    '- "page_span": [page_number]\n\n'
    'Output valid JSON: {"articles": [...]}\n'
    "Do NOT wrap in markdown code blocks. Output raw JSON only."
)

WHOLE_ISSUE_VLM = (
    "You are looking at all pages of a single issue of a historical Italian newspaper. "
    "Transcribe the entire issue and organize it into distinct content units.\n\n"
    "For each content unit (article, advertisement, obituary, notice, editorial), provide:\n"
    '- "unit_type": the type of content\n'
    '- "headline": the headline if present, or null\n'
    '- "paragraphs": list of {{"text": "paragraph text"}}\n'
    '- "page_span": list of page numbers (1-indexed) where this unit appears\n\n'
    'Output valid JSON with a single key "articles" containing these units '
    "ordered by position in the newspaper (front page first, left to right, top to bottom).\n"
    "Preserve the original Italian text exactly as printed."
)


VLM_OCR_ADS_FOCUSED = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "This image shows columns of dense Italian text in old typefaces.\n\n"
    "CRITICAL RULES:\n"
    "1. Read each column TOP-TO-BOTTOM before moving to the next column LEFT-TO-RIGHT\n"
    "2. Transcribe ALL text — especially small ads, classifieds, and footer notices\n"
    "3. Every distinct ad (even 1-line product mentions like 'Tabacco MIL', 'BUTON cognac', "
    "'CERCASI apprendisti') is a SEPARATE unit — do NOT merge adjacent ads\n"
    "4. Editorial credits like 'Direttore responsabile', 'Proprietà letteraria riservata' "
    "are SEPARATE notices\n"
    "5. Preserve the original Italian exactly as printed, including archaic spelling\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial" | "classified"\n'
    '- "headline": the COMPLETE headline or null\n'
    '- "text": the COMPLETE transcribed text\n'
    '- "page_span": [page_number]\n\n'
    'Output valid JSON: {"articles": [...]}. Do NOT wrap in markdown.'
)


VLM_OCR_PAGE_PAIR = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "You are shown TWO CONSECUTIVE PAGES of a single newspaper issue.\n\n"
    "CRITICAL RULES:\n"
    "1. Transcribe each column TOP-TO-BOTTOM, left-to-right, across both pages\n"
    "2. Articles that START on page 1 and CONTINUE on page 2 must be output as ONE unit with page_span=[1,2]\n"
    "3. Articles entirely on page 1: page_span=[1]. Articles entirely on page 2: page_span=[2]\n"
    "4. Transcribe ALL text — do not skip or summarize\n"
    "5. Preserve the original Italian exactly as printed, including archaic spelling\n"
    "6. Separate distinct content units (articles, ads, obituaries, notices)\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial"\n'
    '- "headline": the complete headline text or null\n'
    '- "text": the COMPLETE transcribed text\n'
    '- "page_span": [1] or [2] or [1,2]\n\n'
    'Output valid JSON: {"articles": [...]}. Do NOT wrap in markdown.'
)


VLM_OCR_STRUCTURED_V4_CROSSPAGE = (
    "You are an expert OCR system for historical Italian newspapers (1880-1945). "
    "This image shows columns of dense Italian text in old typefaces.\n\n"
    "CRITICAL RULES:\n"
    "1. Read each column TOP-TO-BOTTOM before moving to the next column LEFT-TO-RIGHT\n"
    "2. Transcribe ALL text — do not skip or summarize anything\n"
    "3. Preserve the original Italian exactly as printed, including archaic spelling\n"
    "4. Separate distinct content units (articles, ads, obituaries, notices)\n"
    "5. If an article ends mid-sentence (no final period) AT THE BOTTOM OF A COLUMN, it likely continues\n"
    "   on the next column or next page — still output what you can see, without inventing text\n\n"
    "For each content unit provide:\n"
    '- "unit_type": "article" | "advertisement" | "obituary" | "notice" | "editorial"\n'
    '- "headline": the COMPLETE headline text or null\n'
    '- "text": the COMPLETE transcribed text (do not truncate)\n'
    '- "page_span": [page_number]\n'
    '- "continues": true if the article text is clearly truncated at the end (ends mid-sentence, '
    'mid-word, or with a dash/ellipsis), false otherwise\n\n'
    'Output valid JSON: {"articles": [...]}\n'
    "Do NOT wrap in markdown code blocks. Output raw JSON only."
)
