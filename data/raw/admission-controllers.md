# Admission Controllers
Admission controllers are plugins that intercept authenticated API requests prior to object persistence in etcd, but after authorization.
They can be mutating (modifying incoming objects, like injecting sidecars) or validating (enforcing security rules and rejecting invalid requests).
