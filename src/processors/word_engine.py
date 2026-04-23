from src.utils.logger import logger
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus, InputFormat
from pathlib import Path
import shutil
import docx2txt
from llama_index.core import Document as LlamaDocument


class WordEngine:
    """Handles DOCX-to-text conversion using Docling (preserving tables and heading structures)
    and falls back to docx2txt for simple text extraction when Docling fails.
    """

    def __init__(self):
        """Initialize the Word converter."""
        # Cấu hình Docling chỉ tập trung xử lý định dạng Word
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.DOCX]
        )
        logger.debug("WordEngine initialized for DOCX processing")

    def normalize_docx_path(self, docx_path: Path) -> Path:
        """Copy the DOCX to a safe temporary path to avoid filename encoding issues.

        Args:
            docx_path: Original path to the DOCX file.

        Returns:
            Path to the copied file (always named "temp_word.docx").
        """
        docx_dir = docx_path.parent
        safe_path = docx_dir / "temp_word.docx"
        shutil.copy(docx_path, safe_path)
        logger.debug(f"Copied '{docx_path.name}' to temporary path: {safe_path}")
        return safe_path

    def extract_text_with_docx2txt(self, file_path: Path) -> str:
        """Extract plain text from a DOCX using docx2txt as a fallback reader.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Extracted text, or an empty string on failure.
        """
        try:
            text = docx2txt.process(str(file_path))
            if text:
                logger.debug(f"docx2txt extracted {len(text)} characters from '{file_path.name}'")
                return text
            return ""
        except Exception as e:
            logger.error(f"Failed to extract text with docx2txt: {e}")
            return ""

    def process_docx(self, docx_path: Path) -> LlamaDocument:
        """Convert a DOCX file to a LlamaIndex Document using Docling, with docx2txt as fallback.

        The source file is first copied to a safe temp path to prevent issues
        with non-ASCII filenames. The temp file is removed after processing.

        Args:
            docx_path: Path to the source DOCX file.

        Returns:
            A llama_index.core.Document object.
            On error, returns an empty Document with error metadata.

        Raises:
            FileNotFoundError: If the DOCX file does not exist.
        """
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        logger.info(f"Processing DOCX: {docx_path.name}")
        safe_path = self.normalize_docx_path(docx_path)

        try:
            result = self.converter.convert(safe_path)

            if result.status == ConversionStatus.SUCCESS:
                # Xuất ra Markdown để LlamaIndex dễ dàng chunking dựa trên Header (#) và Table
                markdown_content = result.document.export_to_markdown()
                logger.info(
                    f"Docling conversion successful — {len(markdown_content)} chars extracted"
                )
                safe_path.unlink(missing_ok=True)
                return LlamaDocument(
                    text=markdown_content,
                    metadata={
                        "source": docx_path.name,
                        "method": "docling",
                        "status": "success",
                    },
                )

            else:
                # Fallback về docx2txt nếu Docling thất bại
                logger.warning(
                    f"Docling conversion failed (status: {result.status}) — falling back to docx2txt"
                )
                text = self.extract_text_with_docx2txt(safe_path)
                safe_path.unlink(missing_ok=True)
                return LlamaDocument(
                    text=text,
                    metadata={
                        "source": docx_path.name,
                        "method": "docx2txt",
                        "status": "success",
                    },
                )

        except Exception as e:
            logger.error(f"Unexpected error while processing DOCX '{docx_path.name}': {e}")
            safe_path.unlink(missing_ok=True)
            return LlamaDocument(
                text="",
                metadata={
                    "source": docx_path.name,
                    "status": "error",
                    "error": str(e),
                },
            )