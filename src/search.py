"""
All interaction with the live web search provider lives here, and only here.
Using DuckDuckGo means no search API key or billing account is required --
useful for an assignment / demo project, though it's a good candidate to
swap for a paid provider (Tavily, SerpAPI) if you need higher reliability
in production.
"""

from duckduckgo_search import DDGS

from src.config import WEB_SEARCH_RESULTS


def get_live_news_context(sport_name, max_results=WEB_SEARCH_RESULTS):
    """
    Searches the live web for recent news, matches, or events for a sport.
    Returns a unified text block of the top search snippets, or a clear
    fallback string if the search fails (so the rest of the pipeline can
    keep running on ChromaDB facts alone rather than crashing).
    """
    search_query = f"{sport_name} latest tournament results championship winners news 2026"
    retrieved_texts = []

    print(f"Executing web search for: '{search_query}'...")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=max_results)
            for index, r in enumerate(results, start=1):
                title = r.get("title", "No title")
                snippet = r.get("body", "No snippet content available")
                retrieved_texts.append(f"Web Source {index}: {title}\nSnippet: {snippet}")
    except Exception as e:
        print(f"Web search failed: {e}")
        return "No recent search engine updates available due to a search error.", []

    if not retrieved_texts:
        return "No recent search engine updates were found.", []

    return "\n\n".join(retrieved_texts), retrieved_texts
