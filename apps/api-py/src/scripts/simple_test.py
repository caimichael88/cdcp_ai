import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.scraper_service import WebScraperService

scraper = WebScraperService(
    base_urls=["https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"],
    max_pages=10,  # Allow more pages
    allowed_paths=['/dental-care-plan/']  # Only scrape CDCP pages
)
documents = scraper.crawl()  # ← Scrapes from URLs!

print(f"\n{'='*60}")
print(f"Scraping complete!")
print(f"Total documents: {len(documents)}")
print(f"{'='*60}\n")

# Show stats
stats = scraper.get_stats()
print("Statistics:")
print(f"  Pages scraped: {stats['pages_scraped']}")
print(f"  Pages failed: {stats['pages_failed']}")
print(f"  Success rate: {stats['success_rate']:.1%}")
print(f"  Retries attempted: {stats.get('retries_attempted', 0)}")
print(f"  Total time: {stats['total_time']:.2f}s")
print(f"  Visited URLs: {stats['visited_urls']}")
print()

print("All scraped pages:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc.title}")
    print(f"   URL: {doc.url}")
    print(f"   Section: {doc.section} | Length: {len(doc.content)} chars")
    print()