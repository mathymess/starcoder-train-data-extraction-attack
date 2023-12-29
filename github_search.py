import requests
import os


def get_search_url(query: str) -> str:
    url = f'https://api.github.com/search/code?q="{query}"'
    return url


def get_headers_with_auth() -> dict[str, str]:
    token = os.getenv("GITHUB_API_TOKEN", default="").strip()
    if len(token) == 0:
        raise ValueError("Environment variable GITHUB_API_TOKEN is not set or is empty")

    headers = {"Authorization": f"Token {token}"}
    return headers


def search_github(code_chunk: str, raise_httperror: bool = True) -> dict:
    """
    Search all GitHub repositories for a code chunk using GitHub API.
    GitHub API Token is read from GITHUB_API_TOKEN environment variable.
    Raise ValueError if the environment variable is not set or is empty.
    The matching is exact (using "").
    Optionally raise HTTPError when the request is not successful.
    Return the response content in JSON format (Python dict).
    """
    url = get_search_url(code_chunk.strip())
    headers = get_headers_with_auth()

    response = requests.request("GET", url, headers=headers)
    if raise_httperror:
        response.raise_for_status()

    return response.json()


def get_gh_occurence_count(code_chunk: str) -> int:
    return search_github(code_chunk)["total_count"]


if __name__ == "__main__":
    """
    Example usage
    """
    code = """
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    """.strip()
    response_json = search_github(code)
    print(f"total count: {response_json['total_count']}")
