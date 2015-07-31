import os

files = os.listdir("../images")
with open("../net_sources/images.txt", 'w') as out:
    path = "../images/"
    for file in files:
        string = "{0} {1}".format(file, 0)
        print string
        out.write(string + '\n')