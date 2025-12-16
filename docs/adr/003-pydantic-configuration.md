# ADR-003: Pydantic for Configuration Management

## Status
Accepted

## Context
The original implementation used scattered `os.getenv()` calls throughout the codebase, leading to:
- No type safety for configuration values
- Difficult to test with different configurations
- No validation of configuration values
- Duplication of default values across services
- Unclear what configuration options are available

A unified configuration system was needed to improve maintainability and reliability.

## Decision
Implement a centralized configuration system using Pydantic Settings with the following structure:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    kafka: KafkaSettings
    database: DatabaseSettings
    # ... other settings
    
    class Config:
        env_file = '.env'
```

**Key Features:**
- Type-safe configuration with runtime validation
- Automatic environment variable parsing
- Nested configuration structure
- Default values with constraints
- Single source of truth in `config/settings.py`

## Consequences

**Positive:**
- **Type Safety:** Pydantic validates types at runtime, catching configuration errors early
- **Validation:** Field validators ensure values are within acceptable ranges
- **Testability:** Easy to override settings for testing
- **Documentation:** Configuration schema is self-documenting
- **IDE Support:** Autocomplete and type hints improve developer experience
- **Industry Standard:** Pydantic is widely used (FastAPI, many production systems)

**Negative:**
- **Migration Effort:** Requires updating all services to use new configuration system
- **Dependency:** Adds pydantic and pydantic-settings as dependencies
- **Learning Curve:** Team needs to understand Pydantic validation

## Alternatives Considered

### Environment Variables Only
- **Pros:** Simple, no dependencies, standard practice
- **Cons:** No type safety, no validation, scattered throughout code
- **Rejected:** Lacks the safety and maintainability needed for production

### Python ConfigParser
- **Pros:** Standard library, no dependencies
- **Cons:** No type safety, requires separate .ini files, less flexible
- **Rejected:** Inferior developer experience compared to Pydantic

### Hydra (Facebook)
- **Pros:** Powerful composition, hierarchical configs
- **Cons:** Overkill for this project, steeper learning curve
- **Rejected:** Too complex for our needs

### Dynaconf
- **Pros:** Multi-environment support, various file formats
- **Cons:** Less type safety than Pydantic, smaller ecosystem
- **Rejected:** Pydantic provides better type safety and validation
