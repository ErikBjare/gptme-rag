# API Documentation Search Example

This example demonstrates how to use gptme-rag to create a searchable index of API documentation, including OpenAPI/Swagger specifications and additional API guides.

## Features Demonstrated

- OpenAPI/Swagger documentation indexing
- API reference searching
- Example request/response lookup
- Endpoint discovery

## Project Structure

```plaintext
api-search/
├── docs/
│   ├── openapi.yaml        # OpenAPI specification
│   ├── guides/
│   │   ├── authentication.md
│   │   ├── rate-limits.md
│   │   └── versioning.md
│   └── examples/
│       ├── requests/
│       └── responses/
├── search_api.py          # Search script
└── index_api.py          # Indexing script
```

## Usage

1. Install gptme-rag:
```bash
pip install gptme-rag
```

2. Index the API documentation:
```bash
python index_api.py
```

3. Search the documentation:
```bash
# Basic search
python search_api.py "how to authenticate"

# Find endpoint
python search_api.py --endpoint "/users"

# Search examples
python search_api.py --examples "create user"
```

## Example Queries

### Finding Endpoints
- "How do I create a new user?"
- "What endpoints handle authentication?"
- "Show me all POST endpoints"

### Authentication
- "How to get an API key?"
- "What authentication methods are supported?"
- "Where to include the access token?"

### Error Handling
- "What does error 429 mean?"
- "How to handle rate limiting?"
- "Show me error response examples"

### Examples
- "Show me a user creation example"
- "What's the response format for GET /users?"
- "Give me a pagination example"

## Customization

### Adding Documentation

1. Update OpenAPI specification in `docs/openapi.yaml`
2. Add markdown guides in `docs/guides/`
3. Include examples in `docs/examples/`

### Search Configuration

Modify search parameters in `search_api.py`:
```python
# Adjust relevance settings
indexer.search(
    query,
    n_results=5,
    where={"type": "endpoint"},  # Filter by doc type
)
```

## Learn More

- See the [gptme-rag documentation](https://github.com/ErikBjare/gptme-rag)
- Explore other examples in the `examples/` directory
- Check out the [OpenAPI guide](https://gptme.org/docs/openapi.html)
