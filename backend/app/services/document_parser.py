
from io import BytesIO

import fitz
from docx import Document


def extract_pdf_text(content: bytes) -> str:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")

            text = "\n".join(
                page.get_text("text")
                for page in pdf
            )

        return text.strip()

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unable to read PDF document.") from exc


def extract_docx_text(content: bytes) -> str:
    try:
        document = Document(BytesIO(content))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        # Also extract text from tables.
        for table in document.tables:
            for row in table.rows:
                paragraphs.extend(
                    cell.text
                    for cell in row.cells
                    if cell.text.strip()
                )

        return "\n".join(paragraphs).strip()

    except Exception as exc:
        raise ValueError("Unable to read DOCX document.") from exc


def extract_resume_text(filename: str, content: bytes) -> str:

    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF file.")

        text = extract_pdf_text(content)

    elif extension == "docx":
        if not content.startswith(b"PK"):
            raise ValueError("Invalid DOCX file.")

        text = extract_docx_text(content)

    else:
        raise ValueError("Only PDF and DOCX files are supported.")

    if len(text) < 50:
        raise ValueError(
            "Insufficient text extracted. The PDF may be scanned or image-based."
        )

    return text
