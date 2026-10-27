# Service Accounts
A ServiceAccount provides an identity for processes that run in a Pod.
When processes inside pods communicate with the apiserver, they authenticate using tokens mounted from their assigned ServiceAccount.
ServiceAccounts can be bound to RBAC Roles to enforce least-privilege API access for automated cluster workloads.
