import os
for file in os.listdir('configs'):
    os.remove(os.path.join('configs', file))