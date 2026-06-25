"""Azure Key Vault helper for fetching application secrets."""

import logging
import os

from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)


class KeyVaultClient:
    """Retrieves secrets from Azure Key Vault or falls back to environment variables."""

    def __init__(self):
        self.vault_url = os.getenv("AZURE_KEYVAULT_URL")
        self.client = None
        if self.vault_url:
            try:
                credential = DefaultAzureCredential()
                self.client = SecretClient(vault_url=self.vault_url, credential=credential)
                logger.info(f"Initialized KeyVaultClient for {self.vault_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize KeyVaultClient: {e}")

    def get_secret(self, secret_name: str, default: str = "") -> str:
        """Fetch secret from Key Vault, fallback to env var."""
        # Try env var first (allows local dev to override or ACA secret references)
        env_val = os.getenv(secret_name.replace("-", "_").upper())
        if env_val:
            return env_val.strip()

        if not self.client:
            return default

        try:
            # Key Vault secret names typically use hyphens instead of underscores
            kv_name = secret_name.replace("_", "-").lower()
            secret = self.client.get_secret(kv_name)
            return secret.value or default
        except ResourceNotFoundError:
            logger.debug(f"Secret {secret_name} not found in Key Vault.")
            return default
        except ClientAuthenticationError:
            logger.error(
                "Authentication failed connecting to Key Vault. Check Managed Identity permissions."
            )
            return default
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return default


# Singleton instance
kv_client = KeyVaultClient()


def get_secret(secret_name: str, default: str = "") -> str:
    """Helper method to get a secret."""
    return kv_client.get_secret(secret_name, default)
