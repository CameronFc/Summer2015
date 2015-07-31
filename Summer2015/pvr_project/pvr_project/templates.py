from header import dirs

class Tstrings:

    @staticmethod
    def getTemplateString(name):
        str = ""
        with open(dirs.path + dirs.templateDirectory + name + dirs.templateExt, "r") as file:
            for line in file:
                str += (line)
        return str + "\n"
