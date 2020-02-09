class Dataset:
    def __init__(self, source, target):
        self.source_lang = source.file_pass
        self.target_lang = target.file_pass

        with open(self.source_lang, 'r') as f:
            self.source_lines = f.readlines()

        with open(self.target_lang, 'r') as f:
            self.target_lines = f.readlines()

    def load(self):
        for line in zip(self.source_lines, self.target_lines):
            print(line[0])
            exit()

        return line

