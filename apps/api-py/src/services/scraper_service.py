"""
Web Scraper Service
Handles scraping CDCP websites and preparing documents for ingestion
"""

import logging
import hashlib
import time
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import tiktoken

try:
    import pysbd
    PYSBD_AVAILABLE = True
except ImportError:
    PYSBD_AVAILABLE = False
    logger.warning("PySBD not available, falling back to regex sentence splitting")

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document"""
    content: str
    url: str
    title: str
    doc_id: str
    section: Optional[str] = None
    last_updated: Optional[str] = None
    language: str = "en"


class WebScraperService:
    """
    Service for scraping web content from CDCP websites
    Extracts clean text and metadata for RAG ingestion
    """
    
    def __init__(
        self,
        base_urls: List[str],
        delay: float = 1.0,
        max_pages: int = 100,
        max_retries: int = 3,
        timeout: int = 30,
        allowed_paths: Optional[List[str]] = None
    ):
        """
        Initialize Web Scraper Service

        Args:
            base_urls: List of starting URLs to scrape
            delay: Delay between requests in seconds (be respectful)
            max_pages: Maximum number of pages to scrape
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds (default: 30)
            allowed_paths: List of URL path patterns to allow (e.g., ['/dental-care-plan/'])
                          If None, all paths are allowed
        """
        self.base_urls = base_urls
        self.delay = delay
        self.max_pages = max_pages
        self.max_retries = max_retries
        self.timeout = timeout
        self.allowed_paths = allowed_paths

        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()

        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CDCP-Chatbot-Scraper/1.0 (Educational/Research Purpose)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
        })

        # Statistics
        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "total_time": 0.0,
            "documents_created": 0,
            "retries_attempted": 0
        }

        logger.info(f"WebScraperService initialized with {len(base_urls)} base URLs")
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid and within allowed domains and paths

        Args:
            url: URL to validate

        Returns:
            True if URL is valid and should be scraped
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False

            # Check if URL is within allowed base domains
            base_domains = [urlparse(base).netloc for base in self.base_urls]
            if parsed.netloc not in base_domains:
                return False

            # Check if URL matches allowed paths (if specified)
            if self.allowed_paths:
                path_matches = any(allowed_path in url for allowed_path in self.allowed_paths)
                if not path_matches:
                    logger.debug(f"URL rejected (not in allowed paths): {url}")
                    return False

            # Skip common non-content URLs
            skip_patterns = [
                r'/search', r'/login', r'/logout', r'/api/',
                r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$',
                r'\.zip$', r'\.doc$', r'\.xlsx?$'
            ]

            if any(re.search(pattern, url, re.I) for pattern in skip_patterns):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return False
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Extract metadata from HTML
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "url": url,
            "scraped_at": datetime.now().isoformat()
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata["description"] = meta_desc['content']
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata["language"] = html_tag['lang'][:2]  # e.g., 'en' from 'en-CA'
        
        # Try to extract section from URL or breadcrumbs
        section = self._extract_section(soup, url)
        if section:
            metadata["section"] = section
        
        # Try to extract last modified date
        modified_meta = soup.find('meta', attrs={'name': 'modified'})
        if modified_meta and modified_meta.get('content'):
            metadata["last_updated"] = modified_meta['content']
        
        return metadata
    
    def _extract_section(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """
        Extract section/category from page
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Section name or None
        """
        # Try breadcrumbs
        breadcrumbs = soup.find('nav', {'aria-label': re.compile('breadcrumb', re.I)})
        if breadcrumbs:
            links = breadcrumbs.find_all('a')
            if links:
                return links[-1].get_text(strip=True).lower()
        
        # Try to extract from URL path
        path = urlparse(url).path
        parts = [p for p in path.split('/') if p and p not in ['en', 'fr', 'services']]
        
        if parts:
            # Common CDCP sections
            section_keywords = {
                'eligibility': 'eligibility',
                'eligible': 'eligibility',
                'coverage': 'coverage',
                'covered': 'coverage',
                'apply': 'application',
                'application': 'application',
                'how-to': 'application',
                'faq': 'faq',
                'questions': 'faq',
                'about': 'overview',
                'overview': 'overview',
                'dental': 'overview'
            }
            
            for part in parts:
                part_lower = part.lower().replace('-', '_')
                if part_lower in section_keywords:
                    return section_keywords[part_lower]
        
        return None
    
    def clean_content(self, soup: BeautifulSoup) -> str:
        """
        Extract and clean main content from HTML
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Clean text content
        """
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                             'iframe', 'noscript', 'aside']):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', {'class': re.compile('content|main', re.I)}) or
            soup.find('body')
        )
        
        if not main_content:
            logger.warning("Could not find main content area")
            return ""
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def scrape_page(self, url: str) -> tuple[Optional[ScrapedDocument], Optional[BeautifulSoup]]:
        """
        Scrape a single page with retry logic

        Args:
            url: URL to scrape

        Returns:
            Tuple of (ScrapedDocument or None, BeautifulSoup or None)
        """
        if url in self.visited_urls:
            return None, None

        if url in self.failed_urls:
            return None, None

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    retry_delay = self.delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retry {attempt}/{self.max_retries} for {url} after {retry_delay:.1f}s")
                    time.sleep(retry_delay)
                    self.stats["retries_attempted"] += 1
                else:
                    logger.info(f"Scraping: {url}")

                # Make request with configurable timeout
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Mark as visited
                self.visited_urls.add(url)

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract metadata
                metadata = self.extract_metadata(soup, url)

                # Clean content
                content = self.clean_content(soup)

                # Validate content
                if len(content) < 100:
                    logger.warning(f"Content too short for {url}, skipping")
                    return None, soup  # Still return soup for link extraction

                # Create document ID
                doc_id = hashlib.md5(url.encode()).hexdigest()

                # Create ScrapedDocument
                document = ScrapedDocument(
                    content=content,
                    url=url,
                    title=metadata.get('title', 'Untitled'),
                    doc_id=doc_id,
                    section=metadata.get('section'),
                    last_updated=metadata.get('last_updated'),
                    language=metadata.get('language', 'en')
                )

                # Update statistics
                self.stats["pages_scraped"] += 1
                self.stats["documents_created"] += 1

                # Respectful delay
                time.sleep(self.delay)

                logger.info(f"✓ Scraped: {metadata.get('title', url)}")
                return document, soup

            except requests.Timeout as e:
                if attempt == self.max_retries - 1:
                    # Final attempt failed due to timeout
                    logger.error(f"Timeout for {url} after {self.max_retries} attempts ({self.timeout}s timeout)")
                    self.failed_urls.add(url)
                    self.stats["pages_failed"] += 1
                    time.sleep(self.delay)  # Delay even on failure
                    return None, None
                else:
                    # Will retry after timeout
                    logger.warning(f"Timeout for {url} (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    continue

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    logger.error(f"Request error for {url} after {self.max_retries} attempts: {e}")
                    self.failed_urls.add(url)
                    self.stats["pages_failed"] += 1
                    time.sleep(self.delay)  # Delay even on failure
                    return None, None
                else:
                    # Will retry
                    logger.warning(f"Request error for {url} (attempt {attempt + 1}): {e}")
                    continue

            except Exception as e:
                # Non-retryable error
                logger.error(f"Error scraping {url}: {e}")
                self.failed_urls.add(url)
                self.stats["pages_failed"] += 1
                time.sleep(self.delay)  # Delay even on failure
                return None, None

        return None, None
    
    def find_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find all valid links on a page
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of valid URLs
        """
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            full_url = full_url.split('#')[0]
            
            # Validate and add
            if (self.is_valid_url(full_url) and 
                full_url not in self.visited_urls and
                full_url not in self.failed_urls):
                links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def crawl(self) -> List[ScrapedDocument]:
        """
        Crawl websites starting from base URLs

        Returns:
            List of ScrapedDocument objects
        """
        start_time = time.time()
        documents = []
        to_visit = list(self.base_urls)

        logger.info(f"Starting crawl with {len(to_visit)} base URLs")
        logger.info(f"Max pages: {self.max_pages}")

        while to_visit and len(documents) < self.max_pages:
            url = to_visit.pop(0)

            if url in self.visited_urls:
                continue

            # Scrape page and get soup (no duplicate request!)
            document, soup = self.scrape_page(url)
            if document:
                documents.append(document)

            # Find more links using the already-fetched soup
            if soup and len(documents) < self.max_pages:
                try:
                    new_links = self.find_links(soup, url)
                    to_visit.extend(new_links)

                except Exception as e:
                    logger.error(f"Error finding links on {url}: {e}")

            # Progress update
            if len(documents) % 10 == 0:
                logger.info(f"Progress: {len(documents)}/{self.max_pages} documents")

        # Final statistics
        self.stats["total_time"] = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"Crawl complete!")
        logger.info(f"  - Documents created: {len(documents)}")
        logger.info(f"  - Pages scraped: {self.stats['pages_scraped']}")
        logger.info(f"  - Pages failed: {self.stats['pages_failed']}")
        logger.info(f"  - Total time: {self.stats['total_time']:.2f}s")
        logger.info("=" * 60)

        return documents
    
    def get_stats(self) -> Dict:
        """Get scraping statistics"""
        return {
            **self.stats,
            "visited_urls": len(self.visited_urls),
            "failed_urls": len(self.failed_urls),
            "success_rate": (
                self.stats["pages_scraped"] / 
                (self.stats["pages_scraped"] + self.stats["pages_failed"])
                if (self.stats["pages_scraped"] + self.stats["pages_failed"]) > 0
                else 0
            )
        }


class DocumentChunker:
    """
    Chunks documents into smaller pieces for better RAG performance
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize Document Chunker

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Number of tokens to overlap between chunks
            model: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model(model)

        # Initialize sentence segmenter if available
        if PYSBD_AVAILABLE:
            self.segmenter = pysbd.Segmenter(language="en", clean=False)
            logger.info(f"DocumentChunker initialized with PySBD (size: {chunk_size}, overlap: {overlap})")
        else:
            self.segmenter = None
            logger.info(f"DocumentChunker initialized with regex fallback (size: {chunk_size}, overlap: {overlap})")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using PySBD or regex fallback

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if self.segmenter:
            # Use PySBD for accurate sentence splitting
            return self.segmenter.segment(text)
        else:
            # Fallback to regex (less accurate)
            return re.split(r'(?<=[.!?])\s+', text)

    def chunk_document(self, document: ScrapedDocument) -> List[ScrapedDocument]:
        """
        Chunk a document into smaller pieces with proper token-based overlap

        Args:
            document: ScrapedDocument to chunk

        Returns:
            List of chunked ScrapedDocument objects
        """
        # Split by sentences using PySBD or regex fallback
        sentences = self.split_sentences(document.content)

        # Pre-calculate token counts for all sentences
        sentence_tokens = [self.count_tokens(s) for s in sentences]

        chunks = []
        current_chunk = []
        current_chunk_tokens = []  # Track individual sentence token counts
        current_tokens = 0

        for sentence, tokens in zip(sentences, sentence_tokens):
            # If adding this sentence exceeds chunk size, save current chunk
            if current_tokens + tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Calculate overlap based on actual overlap parameter
                overlap_sentences = []
                overlap_tokens = 0

                # Work backwards from the end to meet overlap target
                for i in range(len(current_chunk) - 1, -1, -1):
                    if overlap_tokens + current_chunk_tokens[i] <= self.overlap:
                        overlap_sentences.insert(0, current_chunk[i])
                        overlap_tokens += current_chunk_tokens[i]
                    else:
                        break

                # Reset current chunk with overlap
                current_chunk = overlap_sentences
                current_chunk_tokens = [self.count_tokens(s) for s in overlap_sentences]
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_chunk_tokens.append(tokens)
            current_tokens += tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Create ScrapedDocument for each chunk
        chunked_documents = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:  # Skip very small chunks
                continue

            chunked_doc = ScrapedDocument(
                content=chunk_text,
                url=document.url,
                title=document.title,
                doc_id=f"{document.doc_id}_chunk_{i}",
                section=document.section,
                last_updated=document.last_updated,
                language=document.language
            )
            chunked_documents.append(chunked_doc)

        logger.debug(f"Chunked document into {len(chunked_documents)} pieces")
        return chunked_documents