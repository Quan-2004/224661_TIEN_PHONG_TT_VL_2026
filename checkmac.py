import uuid

mac = uuid.getnode()
mac_address = ':'.join(f'{(mac >> i) & 0xff:02x}' for i in range(40, -1, -8))
print(mac_address)