# Secrets
A Secret is an object that contains a small amount of sensitive data such as a password, token, or SSH key.
Secrets decouple sensitive configuration data from container images and pod manifests.
By default, secrets are stored base64-encoded in etcd; clusters in production enable encryption at rest using KMS providers to protect secret data.
