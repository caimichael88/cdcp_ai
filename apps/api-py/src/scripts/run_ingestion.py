"""
Simple script to run CDCP website ingestion
Usage: python run_ingestion.py
"""

import sys
import os
from ingestion_pipeline import CDCPIngestionPipeline

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     CDCP Website Ingestion & RAG System Setup            ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    print("Configuration:")
    print("─" * 60)
    
    # CDCP URLs
    cdcp_urls = [
        "https://www.canada.ca/en/services/benefits/dental/dental-care-plan.html",
        "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/eligibility.html",
        "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/coverage.html",
        "https://www.canada.ca/en/services/benefits/dental/dental-care-plan/apply.html",
    ]
    
    print(f"URLs to scrape: {len(cdcp_urls)}")
    for url in cdcp_urls:
        print(f"  - {url}")
    
    # Settings
    max_pages = int(input("\nMax pages to scrape (default 50): ") or "50")
    chunk_size = int(input("Chunk size in tokens (default 512): ") or "512")
    
    print("\nEmbedding provider options:")
    print("  1. sentence-transformers (free, local, recommended)")
    print("  2. openai (paid, requires API key)")
    provider_choice = input("Choose provider (1 or 2, default 1): ") or "1"
    
    if provider_choice == "2":
        embedding_provider = "openai"
        api_key = input("Enter OpenAI API key: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("No API key provided, falling back to sentence-transformers")
            embedding_provider = "sentence-transformers"
    else:
        embedding_provider = "sentence-transformers"
    
    collection_name = input("\nCollection name (default 'cdcp_documents'): ") or "cdcp_documents"
    
    # Confirmation
    print("\n" + "─" * 60)
    print("Pipeline Configuration:")
    print(f"  - URLs: {len(cdcp_urls)}")
    print(f"  - Max pages: {max_pages}")
    print(f"  - Chunk size: {chunk_size} tokens")
    print(f"  - Embedding provider: {embedding_provider}")
    print(f"  - Collection name: {collection_name}")
    print("─" * 60)
    
    confirm = input("\nProceed with ingestion? (y/n): ").lower()
    if confirm != 'y':
        print("Ingestion cancelled.")
        return
    
    # Create pipeline
    print("\n" + "=" * 60)
    print("Starting Ingestion Pipeline...")
    print("=" * 60 + "\n")
    
    try:
        pipeline = CDCPIngestionPipeline(
            cdcp_urls=cdcp_urls,
            max_pages=max_pages,
            chunk_size=chunk_size,
            embedding_provider=embedding_provider,
            collection_name=collection_name
        )
        
        # Run pipeline
        results = pipeline.run()
        
        if results["success"]:
            # Test the system
            print("\n" + "=" * 60)
            test = input("Would you like to test the RAG system? (y/n): ").lower()
            if test == 'y':
                pipeline.test_rag_system()
            
            # Summary
            print("\n" + "╔" + "═" * 58 + "╗")
            print("║" + " " * 20 + "SUCCESS!" + " " * 27 + "║")
            print("╚" + "═" * 58 + "╝")
            print(f"\n✓ Scraped: {results['scraped_documents']} documents")
            print(f"✓ Chunked: {results['chunked_documents']} chunks")
            print(f"✓ Ingested: {results['ingested_documents']} documents")
            print(f"\nYour RAG system is ready!")
            print(f"Collection: {collection_name}")
            print(f"Documents: {results['rag_stats']['vector_database']['document_count']}")
            
            print("\n" + "─" * 60)
            print("Next Steps:")
            print("─" * 60)
            print("1. Start the API server:")
            print("   python app.py")
            print("\n2. Test with curl:")
            print("   curl -X POST http://localhost:5000/search \\")
            print("     -H 'Content-Type: application/json' \\")
            print("     -d '{\"query\": \"What is CDCP eligibility?\"}'")
            print("\n3. Or test programmatically:")
            print("   python")
            print("   >>> from rag_controller import create_rag_controller")
            print(f"   >>> controller = create_rag_controller(collection_name='{collection_name}')")
            print("   >>> results = controller.search('CDCP eligibility')")
            print("   >>> print(results[0].content)")
            print("─" * 60)
            
        else:
            print("\n" + "╔" + "═" * 58 + "╗")
            print("║" + " " * 22 + "FAILED" + " " * 28 + "║")
            print("╚" + "═" * 58 + "╝")
            print(f"\nError: {results.get('error', 'Unknown error')}")
            print("\nPlease check the logs above for details.")
    
    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 22 + "ERROR" + " " * 29 + "║")
        print("╚" + "═" * 58 + "╝")
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()