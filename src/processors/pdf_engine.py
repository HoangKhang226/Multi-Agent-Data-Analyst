from src.utils.logger import logger
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
import shutil
import pypdf
from llama_index.core import Document as LlamaDocument


class PDFEngine:
    """Handles PDF-to-text conversion using Docling (with OCR/table support)
    and falls back to pypdf for simple text extraction when Docling fails.
    """

    def __init__(self):
        """Initialize the PDF converter with OCR and table structure detection enabled."""
        self.pdf_pipeline_options = PdfPipelineOptions()
        self.pdf_pipeline_options.do_table_structure = True
        self.pdf_pipeline_options.do_ocr = True

        self.pdf_format_options = PdfFormatOption(
            pipeline_options=self.pdf_pipeline_options
        )

        # Build the Docling document converter with PDF-specific pipeline options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pdf_pipeline_options
                )
            }
        )
        logger.debug("PDFEngine initialized with OCR and table structure detection enabled")

    def normalize_pdf_path(self, pdf_path: Path) -> Path:
        """Copy the PDF to a safe temporary path to avoid filename encoding issues.

        Args:
            pdf_path: Original path to the PDF file.

        Returns:
            Path to the copied file (always named "temp.pdf").
        """
        pdf_dir = pdf_path.parent
        safe_path = pdf_dir / "temp.pdf"
        shutil.copy(pdf_path, safe_path)
        logger.debug(f"Copied '{pdf_path.name}' to temporary path: {safe_path}")
        return safe_path

    def extract_text_with_pypdf(self, file_path: Path) -> str:
        """Extract plain text from a PDF using pypdf as a fallback reader.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Concatenated text from all pages, or an empty string on failure.
        """
        text = ""
        try:
            with open(file_path, "rb", encoding="utf-8") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            logger.debug(f"pypdf extracted {len(text)} characters from '{file_path.name}'")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text with pypdf: {e}")
            return ""

    def process_pdf(self, pdf_path: Path) -> LlamaDocument:
        """Convert a PDF file to a LlamaIndex Document using Docling, with pypdf as fallback.

        The source file is first copied to a safe temp path to prevent issues
        with non-ASCII filenames. The temp file is removed after processing.

        Args:
            pdf_path: Path to the source PDF file.

        Returns:
            A llama_index.core.Document object.
            On error, returns an empty Document with error metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path.name}")
        safe_path = self.normalize_pdf_path(pdf_path)

        try:
            result = self.converter.convert(safe_path)

            if result.status == ConversionStatus.SUCCESS:
                markdown_content = result.document.export_to_markdown()
                logger.info(
                    f"Docling conversion successful — {len(markdown_content)} chars extracted"
                )
                safe_path.unlink(missing_ok=True)
                return LlamaDocument(
                    text=markdown_content,
                    metadata={
                        "source": pdf_path.name,
                        "method": "docling",
                        "status": "success",
                    },
                )

            else:
                # Docling conversion failed; fall back to basic pypdf text extraction
                logger.warning(
                    f"Docling conversion failed (status: {result.status}) — falling back to pypdf"
                )
                text = self.extract_text_with_pypdf(safe_path)
                safe_path.unlink(missing_ok=True)
                return LlamaDocument(
                    text=text,
                    metadata={
                        "source": pdf_path.name,
                        "method": "pypdf",
                        "status": "success",
                    },
                )

        except Exception as e:
            logger.error(f"Unexpected error while processing PDF '{pdf_path.name}': {e}")
            safe_path.unlink(missing_ok=True)
            return LlamaDocument(
                text="",
                metadata={
                    "source": pdf_path.name,
                    "status": "error",
                    "error": str(e),
                },
            )