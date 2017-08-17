import os, subprocess

# List of currently running nfv apps
NFV_RUNNING = []
NO_CHANGE = 0
L3_DECREASE = 1
L3_INCREASE = 2 

# Hex mask strings (find better way)
MASK = ["0","1","3","7","f"]

# Metadata for each NFV app modify flag says if it needs to be realloced
# 0 - no change
# 1 - decrease
# 2 - increase

class NFV_APP():
    def __init__(self, app, core,cos, targetpr, currentpr, l3alloc, modify=NO_CHANGE):
        self.app = app
        self.core = core
        self.cos = cos
        self.targetpr = targetpr
        self.currentpr = currentpr
        self.l3alloc = l3alloc
        self.modify = modify

    def getCurrentAlloc(self, l3alloc):
        return l3alloc

# For now manually add app paramters for a NAT application
NAT = NFV_APP('nat', '10','1' '14', '-1', '0x1',NO_CHANGE)

# Initailize NAT application paramters
NFV_RUNNING.append(NAT)

ALLOCATION_DIR="/home/shelbyt/research/lambda-balance-nfv/"

args = (ALLOCATION_DIR + "allocation_app", "-h")
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()
print output


#target in MPPS
current_app = 0
nf_target=14
nf_current = -1
nf_core = -1
# 0 if we keep current config
# 1 if we want to change config
nf_modify = 0

line_count = 0

# This packet data is for the NAT NFV app
f = open('nat.dat','r')
for line in f:
    if line_count == 0:
        # first line record core info
        #nf_core=int(line)
        NAT.core = int(line)
    else:
        line_count = 1
        # if we are not meeting the target
        if NAT.targetpr > NAT.currentpr:
            # 2 means we need to increase the alloc
            NAT.modify = L3_INCREASE

if NAT.modify == L3_INCREASE:




