from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageText:
    page_number: int
    text: str


def parse_pdf(path: str) -> list[PageText]:
    doc = fitz.open(path)
    try:
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(PageText(page_number=i + 1, text=text))
        return pages
    finally:
        doc.close()
