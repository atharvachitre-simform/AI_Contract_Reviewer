#!/bin/bash
set -e

# ==============================================================================
# SECURE AZURE CONTAINER APPS DEPLOYMENT SCRIPT
# Architecture: Dual Containers (Frontend UI -> Backend API)
# Security: Key Vault References, Managed Identities, Entra ID EasyAuth, VNet
# ==============================================================================

RESOURCE_GROUP="rg-contract-reviewer-prod"
LOCATION="eastus"
ENV_NAME="cae-contract-reviewer"
VNET_NAME="vnet-contract-reviewer"
KEYVAULT_NAME="kv-contract-prod-001"
ACR_NAME="contractreviewregistry22451"
IDENTITY_NAME="id-contract-reviewer"

echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating VNet and Subnet for Container Apps..."
az network vnet create \
    --resource-group $RESOURCE_GROUP \
    --name $VNET_NAME \
    --address-prefix 10.0.0.0/16 \
    --subnet-name ca-subnet \
    --subnet-prefix 10.0.0.0/23

SUBNET_ID=$(az network vnet subnet show -g $RESOURCE_GROUP -n ca-subnet --vnet-name $VNET_NAME --query id -o tsv)

echo "Creating Managed Identity..."
az identity create -g $RESOURCE_GROUP -n $IDENTITY_NAME
IDENTITY_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query id -o tsv)
IDENTITY_CLIENT_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query clientId -o tsv)
IDENTITY_PRINCIPAL_ID=$(az identity show -g $RESOURCE_GROUP -n $IDENTITY_NAME --query principalId -o tsv)

echo "Creating Key Vault and assigning RBAC..."
az keyvault create -g $RESOURCE_GROUP -n $KEYVAULT_NAME --location $LOCATION --enable-rbac-authorization true
az role assignment create \
    --role "Key Vault Secrets User" \
    --assignee $IDENTITY_PRINCIPAL_ID \
    --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME"

echo "Populating Key Vault with third-party secrets..."
az keyvault secret set --vault-name $KEYVAULT_NAME -n "groq-api-key" --value "$GROQ_API_KEY"
az keyvault secret set --vault-name $KEYVAULT_NAME -n "qdrant-api-key" --value "$QDRANT_API_KEY"
az keyvault secret set --vault-name $KEYVAULT_NAME -n "supabase-password" --value "$SUPABASE_PASSWORD"

echo "Creating Container Apps Environment (Internal VNet)..."
az containerapp env create \
    --name $ENV_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --infrastructure-subnet-resource-id $SUBNET_ID \
    --internal-only false # Environment can handle both internal and external apps

echo "Deploying BACKEND App (Internal Only)..."
az containerapp create \
    --name ca-backend \
    --resource-group $RESOURCE_GROUP \
    --environment $ENV_NAME \
    --image $ACR_NAME.azurecr.io/contract-reviewer-backend:latest \
    --target-port 8000 \
    --ingress internal \
    --user-assigned $IDENTITY_ID \
    --secrets \
        "groq-api-key=keyvaultref:$KEYVAULT_NAME/secrets/groq-api-key,identityref:$IDENTITY_ID" \
        "qdrant-api-key=keyvaultref:$KEYVAULT_NAME/secrets/qdrant-api-key,identityref:$IDENTITY_ID" \
    --env-vars \
        "GROQ_API_KEY=secretref:groq-api-key" \
        "QDRANT_API_KEY=secretref:qdrant-api-key" \
        "AZURE_CLIENT_ID=$IDENTITY_CLIENT_ID" \
        "AZURE_KEYVAULT_URL=https://$KEYVAULT_NAME.vault.azure.net/"

BACKEND_FQDN=$(az containerapp show -n ca-backend -g $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)

echo "Deploying FRONTEND App (External Ingress)..."
az containerapp create \
    --name ca-frontend \
    --resource-group $RESOURCE_GROUP \
    --environment $ENV_NAME \
    --image $ACR_NAME.azurecr.io/contract-reviewer-frontend:latest \
    --target-port 8501 \
    --ingress external \
    --env-vars \
        "BACKEND_URL=https://$BACKEND_FQDN"

echo "Configuring Entra ID (EasyAuth) on Frontend..."
# Requires setting up an App Registration in Microsoft Entra ID
az containerapp auth microsoft update \
    --name ca-frontend \
    --resource-group $RESOURCE_GROUP \
    --client-id $ENTRA_APP_CLIENT_ID \
    --client-secret $ENTRA_APP_CLIENT_SECRET \
    --tenant-id $ENTRA_TENANT_ID \
    --yes

az containerapp auth update \
    --name ca-frontend \
    --resource-group $RESOURCE_GROUP \
    --action Return401 \
    --enabled true

echo "Deployment Securely Completed."
