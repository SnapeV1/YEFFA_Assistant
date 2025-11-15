# YEFFA_Assistant

### Running the API from a VM

1. Start the FastAPI server inside the VM with `python api.py` (or Docker). The application already binds to `0.0.0.0:5001`, so it listens on every VM interface.
2. From your host machine (where the Angular UI runs), call the API by targeting the VM address directly, for example `http://192.168.182.128:5001/respond`.
3. CORS is wide open (allows all origins), so no extra configuration is needed to talk to the API across host/VM boundaries.
4. Every HTTP request/response is logged (method, path, client IP, status, duration) which makes debugging host↔VM connectivity straightforward—watch the VM console while the host frontend calls the API.
5. Make sure the VM firewall allows inbound TCP/5001 traffic and, if you are using NAT/Bridged networking, that the host and VM are on the same virtual network.
