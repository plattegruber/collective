terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.30"
    }
  }
}

provider "cloudflare" {
  api_token = var.api_token
}

variable "api_token" {
  type = string
}

variable "account_id" {
  type = string
}

resource "cloudflare_workers_kv_namespace" "reactions" {
  account_id = var.account_id
  title      = "gg-reactions"
}

output "kv_namespace_id" {
  value = cloudflare_workers_kv_namespace.reactions.id
}
