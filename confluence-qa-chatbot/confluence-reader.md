## Using the Atlassian Python Package for Recursive Queries

This guide outlines how to use the `atlassian-python-api` package for performing recursive queries against a Confluence instance.

### Installation

To begin, install the package using pip:

```bash
pip install atlassian-python-api
```

### Recursive Query Example

Here's an example of how to recursively fetch pages starting from a specific page ID:

```python
from atlassian import Confluence
from pprint import pprint

# Initialize the Confluence client
confluence = Confluence(
    url='https://your-domain.atlassian.net',
    username='your-email@example.com',
    password='your-api-token',  # Or use an API token
    cloud=True  # Set to False for self-hosted/Data Center
)

def get_page_hierarchy(page_id, level=0, max_depth=None):
    """
    Recursively fetches page hierarchy starting from a given page ID
    
    Args:
        page_id: The starting page ID
        level: Current depth level (used for indentation)
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    if max_depth is not None and level > max_depth:
        return
    
    # Get the page details
    page = confluence.get_page_by_id(page_id, expand='body.storage,children.page')
    
    # Print page info with indentation based on level
    print(f"{'  ' * level}📄 {page['title']} (ID: {page_id})")
    
    # Get child pages
    children = confluence.get_page_child_by_type(page_id, type='page', start=0, limit=50)
    
    # Recursively process each child
    for child in children:
        get_page_hierarchy(child['id'], level + 1, max_depth)

# Example usage
start_page_id = '123456'  # Replace with your starting page ID
get_page_hierarchy(start_page_id, max_depth=5)  # Limit to 5 levels deep
```

### Key Features of the Atlassian Package

The `atlassian-python-api` package offers several features that simplify interaction with Confluence:

*   **Built-in Pagination Handling:** The package automatically handles pagination for methods like `get_page_child_by_type`, simplifying the retrieval of large datasets.
*   **Expanded Methods:** The package provides a range of methods for interacting with Confluence pages, including:
    *   `get_page_by_id()`: Retrieves page content.
    *   `get_page_child_by_type()`: Retrieves child pages.
    *   `get_descendant_page_ids()`: Retrieves all descendant page IDs (non-recursive).
    *   `get_page_ancestors()`: Retrieves page ancestors.

### Additional Useful Methods

Beyond the core methods, the package offers additional functionality:

*   **CQL Search:** You can search pages using Confluence Query Language (CQL).
    ```python
    # Search pages using CQL
    results = confluence.cql('ancestor=123456', limit=100)
    ```
*   **Retrieve All Pages in a Space:**  You can retrieve all pages within a specific Confluence space.
    ```python
    # Get all pages in a space
    all_space_pages = confluence.get_all_pages_from_space('YOURSPACE')
    ```

### Advanced Example with Data Collection

To collect all page data in a structured format, you can use the following approach:

```python
def collect_page_hierarchy(page_id, hierarchy=None):
    if hierarchy is None:
        hierarchy = []
    
    page = confluence.get_page_by_id(page_id)
    page_data = {
        'id': page_id,
        'title': page['title'],
        'children': []
    }
    
    children = confluence.get_page_child_by_type(page_id, type='page')
    for child in children:
        child_data = collect_page_hierarchy(child['id'])
        page_data['children'].append(child_data)
    
    hierarchy.append(page_data)
    return hierarchy

# Usage
hierarchy = collect_page_hierarchy('123456')
pprint(hierarchy)
```

### Error Handling

For production environments, it's crucial to implement proper error handling:

```python
from atlassian.errors import ApiError

try:
    get_page_hierarchy('123456')
except ApiError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

The `atlassian-python-api` package simplifies working with Confluence's API and handles many low-level details, making it a good choice for recursive queries and other Confluence operations.