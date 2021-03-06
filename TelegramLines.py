class TelegramLines():

    def __init__(self):
        self.textLines = []

    # return the textlines in their string locations separated by newlines
    def __str__(self):
        return "\n".join([textLine.__str__() for textLine in self.textLines])

    def addLine(self, textLine):
        self.textLines.append(textLine)

    def getLines(self):
        return self.textLines

    def getNumLines(self):
        return len(self.textLines)
