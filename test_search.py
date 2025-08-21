from tools.search import (
    get_search_manager,
    get_content_fetcher,
    get_search_tool,
    search_and_fetch,
    search_for_topic
)

def main():
    # Test: Search Manager
    search_manager = get_search_manager()
    results = search_manager.search("AI in Healthcare", max_results=3)
    print("\n=== Search Manager Results ===")
    for r in results:
        print(r.to_dict())

    # Test: Content Fetcher
    content_fetcher = get_content_fetcher()
    if results and results[0].url:
        content = content_fetcher.fetch_content(results[0].url)
        print("\n=== Content Fetcher Test ===")
        print(content[:500])  # Print first 500 chars

    # Test: LangChain Tool Wrapper
    search_tool = get_search_tool()
    formatted_results = search_tool._run("AI in Healthcare", max_results=2)
    print("\n=== ResearchSearchTool Results ===")
    print(formatted_results)

    # Test: Combined Search + Fetch
    enriched = search_and_fetch("AI in Healthcare", max_results=3)
    print("\n=== Search + Fetch Enriched Results ===")
    for e in enriched:
        print(e["title"], "| content length:", e["content_length"])

    # Test: Topic Search with Depth
    depth_results = search_for_topic("AI in Healthcare", depth=3)
    print("\n=== Search for Topic (Depth=3) ===")
    for d in depth_results:
        print(d["title"], "|", d["url"])

if __name__ == "__main__":
    main()
