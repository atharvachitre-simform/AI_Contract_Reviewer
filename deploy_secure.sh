#!/bin/bash
set -e

# ==============================================================================
# SECURE AZURE CONTAINER APPS DEPLOYMENT SCRIPT (TAILORED FOR PRODUCTION)
# Resource Group: AIContractReviewerRG
# Environment: contract-reviewer-env
# Architecture: Dual Containers (Frontend UI -> Backend API)
# Security: Key Vault References, Managed Identities, VNet-equivalent internal ingress
# ==============================================================================

RESOURCE_GROUP="AIContractReviewerRG"
LOCATION="eastus"
ENV_NAME="contract-reviewer-env"
ACR_NAME="contractreviewregistry22451"
IDENTITY_NAME="id-contract-reviewer"
KEYVAULT_NAME="kv-reviewer-22451"
BACKEND_APP="backend"
FRONTEND_APP="frontend"

# Default to the latest built image tag that passed the Trivy scan
IMAGE_TAG="a96e1e87a260f8c58244c36fa86c762c4322b218"

echo "=== Loading environment variables from .env ==="
if [ -f .env ]; then
    # Export non-commented environment variables
    export $(grep -v '^#' .env | xargs)
    # Parse Supabase password from connection string if not explicitly defined
    if [ -z "$SUPABASE_PASSWORD" ] && [ -n "$SUPABASE_DIRECT_CONNECTION_STRING" ]; then
        SUPABASE_PASSWORD=$(echo "$SUPABASE_DIRECT_CONNECTION_STRING" | sed -E 's/.*:\/\/.*:(.*)@.*/\1/')
        echo "Parsed SUPABASE_PASSWORD from connection string."
    fi
else
    echo "ERROR: .env file not found in current directory. Please run this script from the project root."
    exit 1
fi

echo "=== Verifying essential environment secrets ==="
if [ -z "$GROQ_API_KEY" ] || [ -z "$QDRANT_API_KEY" ] || [ -z "$SUPABASE_PASSWORD" ]; then
    echo "ERROR: Missing required keys in .env (GROQ_API_KEY, QDRANT_API_KEY, or SUPABASE connection string password)."
    exit 1
fi

echo "=== 1. Checking/Creating Managed Identity ==="
IDENTITY_ID=$(az identity list -g $RESOURCE_GROUP --query "[?name=='$IDENTITY_NAME'].id" -o tsv)
if [ -z "$IDENTITY_ID" ]; then
    echo "Creating User-Assigned Managed Identity '$IDENTITY_NAME'..."
    az identity create -g $RESOURCE_GROUP -n $IDENTITY_NAME
    IDENTITY_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query id -o tsv)
else
    echo "Managed Identity '$IDENTITY_NAME' already exists."
fi

IDENTITY_CLIENT_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query clientId -o tsv)
IDENTITY_PRINCIPAL_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query principalId -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "=== 2. Checking/Creating Key Vault ==="
KV_EXISTS=$(az keyvault list -g $RESOURCE_GROUP --query "[?name=='$KEYVAULT_NAME'].name" -o tsv)
if [ -z "$KV_EXISTS" ]; then
    echo "Creating Key Vault '$KEYVAULT_NAME' with RBAC authorization..."
    az keyvault create -g $RESOURCE_GROUP -n $KEYVAULT_NAME --location $LOCATION --enable-rbac-authorization true
else
    echo "Key Vault '$KEYVAULT_NAME' already exists."
fi

echo "=== 3. Assigning Key Vault Secrets Officer role to Deployer ==="
USER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv)
az role assignment create \
    --role "Key Vault Secrets Officer" \
    --assignee-object-id $USER_OBJECT_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME" \
    --assignee-principal-type User || echo "Deployer role assignment already exists."

echo "=== 4. Assigning Key Vault Secrets User role to Managed Identity ==="
az role assignment create \
    --role "Key Vault Secrets User" \
    --assignee-object-id $IDENTITY_PRINCIPAL_ID \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME" \
    --assignee-principal-type ServicePrincipal || echo "Identity role assignment already exists."

echo "Sleeping 15 seconds to allow Azure RBAC role assignments to propagate..."
sleep 15

echo "=== 5. Storing Secrets in Key Vault ==="
az keyvault secret set --vault-name $KEYVAULT_NAME -n "groq-api-key" --value "$GROQ_API_KEY" > /dev/null
az keyvault secret set --vault-name $KEYVAULT_NAME -n "qdrant-api-key" --value "$QDRANT_API_KEY" > /dev/null
az keyvault secret set --vault-name $KEYVAULT_NAME -n "supabase-password" --value "$SUPABASE_PASSWORD" > /dev/null
echo "Secrets successfully updated in Key Vault."

echo "=== 6. Registering Key Vault access on ACR ==="
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

echo "=== 7. Updating BACKEND Container App (Subcommands) ==="
echo "Assigning Managed Identity to backend..."
az containerapp identity assign \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --user-assigned $IDENTITY_ID

echo "Setting registry login credentials on backend..."
az containerapp registry set \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --server $ACR_NAME.azurecr.io \
    --username $ACR_USERNAME \
    --password $ACR_PASSWORD

echo "Adding Key Vault secret references to backend secrets..."
az containerapp secret set \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --secrets \
        "groq-api-key=keyvaultref:https://$KEYVAULT_NAME.vault.azure.net/secrets/groq-api-key,identityref:$IDENTITY_ID" \
        "qdrant-api-key=keyvaultref:https://$KEYVAULT_NAME.vault.azure.net/secrets/qdrant-api-key,identityref:$IDENTITY_ID" \
        "supabase-password=keyvaultref:https://$KEYVAULT_NAME.vault.azure.net/secrets/supabase-password,identityref:$IDENTITY_ID"

echo "Setting ingress type to internal for backend..."
az containerapp ingress update \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --type internal

echo "Deploying the secure image and setting Key Vault references on backend..."
az containerapp update \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --image $ACR_NAME.azurecr.io/contract-reviewer-backend:$IMAGE_TAG \
    --set-env-vars \
        "GROQ_API_KEY=secretref:groq-api-key" \
        "QDRANT_API_KEY=secretref:qdrant-api-key" \
        "SUPABASE_PASSWORD=secretref:supabase-password" \
        "AZURE_CLIENT_ID=$IDENTITY_CLIENT_ID" \
        "AZURE_KEYVAULT_URL=https://$KEYVAULT_NAME.vault.azure.net/"

# Retrieve internal FQDN of the backend app
BACKEND_FQDN=$(az containerapp show -n $BACKEND_APP -g $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
echo "Backend internal FQDN is: $BACKEND_FQDN"

echo "=== 8. Updating FRONTEND Container App (Subcommands) ==="
echo "Setting registry login credentials on frontend..."
az containerapp registry set \
    --name $FRONTEND_APP \
    --resource-group $RESOURCE_GROUP \
    --server $ACR_NAME.azurecr.io \
    --username $ACR_USERNAME \
    --password $ACR_PASSWORD

echo "Deploying the new image and pointing backend FQDN on frontend..."
az containerapp update \
    --name $FRONTEND_APP \
    --resource-group $RESOURCE_GROUP \
    --image $ACR_NAME.azurecr.io/contract-reviewer-backend:$IMAGE_TAG \
    --command "streamlit" \
    --args "run streamlit_app.py --server.port=8501 --server.address=0.0.0.0" \
    --set-env-vars \
        "BACKEND_URL=https://$BACKEND_FQDN"

echo "=== 9. Configuring Entra ID (EasyAuth) on Frontend (Optional) ==="
if [ -z "$ENTRA_APP_CLIENT_ID" ] || [ -z "$ENTRA_APP_CLIENT_SECRET" ] || [ -z "$ENTRA_TENANT_ID" ]; then
    echo "WARNING: Entra ID credentials (ENTRA_APP_CLIENT_ID, ENTRA_APP_CLIENT_SECRET, ENTRA_TENANT_ID) not found in shell/env."
    echo "Skipping EasyAuth setup. To configure Entra ID authentication later, run:"
    echo "  az containerapp auth microsoft update --name frontend --resource-group $RESOURCE_GROUP \\"
    echo "    --client-id <client-id> --client-secret <client-secret> --tenant-id <tenant-id> --yes"
    echo "  az containerapp auth update --name frontend --resource-group $RESOURCE_GROUP --action Return401 --enabled true"
else
    echo "Configuring Microsoft Entra ID (EasyAuth) on Frontend Container App..."
    az containerapp auth microsoft update \
        --name $FRONTEND_APP \
        --resource-group $RESOURCE_GROUP \
        --client-id $ENTRA_APP_CLIENT_ID \
        --client-secret $ENTRA_APP_CLIENT_SECRET \
        --tenant-id $ENTRA_TENANT_ID \
        --yes
    
    az containerapp auth update \
        --name $FRONTEND_APP \
        --resource-group $RESOURCE_GROUP \
        --action Return401 \
        --enabled true
    echo "Entra ID EasyAuth successfully enabled."
fi

echo "=== Deployment Securely Completed. ==="
echo "Frontend URL: https://$(az containerapp show -n $FRONTEND_APP -g $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)"
