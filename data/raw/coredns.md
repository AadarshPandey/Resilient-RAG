# CoreDNS
CoreDNS is a flexible, extensible DNS server that serves as the default cluster DNS provider in Kubernetes.
It runs as a Deployment inside the kube-system namespace and watches the Kubernetes API for services and endpoints, dynamically updating internal DNS records to enable service discovery by hostname.
