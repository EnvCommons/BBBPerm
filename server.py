from openreward.environments import Server

from bbbperm import BBBPerm

if __name__ == "__main__":
    server = Server([BBBPerm])
    server.run()
