import requests
import urllib
import os
import re


def get_search_url(query: str) -> str:
    query = query.strip()
    # query = re.sub("\s+", " ", query)  # Normalize whitespace characters
    # query = re.sub("\s+", " ", query)  # Remove duplicate spaces
    query = urllib.parse.quote(query)
    url = f'https://api.github.com/search/code?q="{query}"'
    return url


def get_headers_with_auth() -> dict[str, str]:
    token = os.getenv("GITHUB_API_TOKEN", default="").strip()
    if len(token) == 0:
        raise ValueError("Environment variable GITHUB_API_TOKEN is not set or is empty")

    headers = {"Authorization": f"Token {token}"}
    return headers


def search_github(code_chunk: str) -> dict:
    """
    Search all GitHub repositories for a code chunk using GitHub API.
    GitHub API Token is read from GITHUB_API_TOKEN environment variable.
    Raise ValueError if the environment variable is not set or is empty.
    The matching is exact (using "").
    Return the response object
    """
    url = get_search_url(code_chunk.strip())
    headers = get_headers_with_auth()
    return requests.request("GET", url, headers=headers)


def get_gh_occurence_count(code_chunk: str, raise_httperror: bool = True) -> int:
    """
    Raise HTTPError when the request is not successful.
    Return the total occurence count of the code chunk on GitHub from the response JSON.
    """
    response = search_github(code_chunk)
    response.raise_for_status()
    response_json = response.json()
    return response_json["total_count"]


if __name__ == "__main__":
    """
    Example usage
    """
    code = """
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Without this comment, about 300 matches. But this unique comment brings it down to 0.
    """.strip()
    print(get_search_url(code))
    print(f"total count: {get_gh_occurence_count(code)}")
