# Network Policies
A NetworkPolicy is a specification of how groups of Pods are allowed to communicate with each other and with other network endpoints.
NetworkPolicy resources use pod selectors, namespace selectors, and CIDR blocks to define ingress and egress traffic rules.
A network plugin supporting NetworkPolicy enforcement, such as Calico or Cilium, must be installed for policies to take effect.
