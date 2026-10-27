# Pod Security Standards
Kubernetes Pod Security Standards define three levels of security policies: Privileged (unrestricted), Baseline (prevents known privilege escalations), and Restricted (hardened against container breakouts).
The built-in Pod Security Admission controller enforces these standards per namespace using enforce, audit, and warn modes.
