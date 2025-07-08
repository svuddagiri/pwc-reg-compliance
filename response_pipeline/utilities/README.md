# Utilities

This folder contains utility scripts and tools for the Regulatory Query Agent.

## Folders

### environment_checks/
Environment verification scripts to test connections and configurations:

- `check_search_fields.py` - Verify Azure AI Search index fields
- `check_tables.py` - Verify database tables are created correctly
- `test_analytics_service.py` - Test analytics service functionality
- `test_api_endpoints.py` - Test API endpoint connectivity
- `test_azure_openai.py` - Test Azure OpenAI connection
- `test_prompt_templates.py` - Test prompt template loading
- `test_response_generator.py` - Test response generation
- `test_security_components.py` - Test security components

## Usage

These scripts are used during initial environment setup to verify all services are properly configured. See `/docs/environment_setup.md` for detailed usage instructions.

Example:
```bash
# Test database connectivity
python utilities/environment_checks/check_tables.py

# Test Azure OpenAI
python utilities/environment_checks/test_azure_openai.py

# Verify search index
python utilities/environment_checks/check_search_fields.py
```