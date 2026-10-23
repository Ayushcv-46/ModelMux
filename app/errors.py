class ProviderError(Exception):
    """Something went wrong at the provider (parent of all provider errors)."""

    status_code = 502

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderTimeout(ProviderError):
    status_code = 504


class ProviderRateLimited(ProviderError):
    status_code = 429