from sql_metadata import Parser

# build my own parser

class sql_parser(Parser):
    def __init__(self, query):
        super().__init__(query)
        self.cleaned_columns = [col for col in self.columns if self.validate_column(col)]
        self.cleaned_columns_dict = {}
        for k, v in self.columns_dict.items():
            for col in v:
                if self.cleaned_columns_dict.get(k) is None:
                    self.cleaned_columns_dict[k] = []
                if self.validate_column(col):
                    self.cleaned_columns_dict[k].append(col)
    def validate_column(self, column):
        return "." in column and " " not in column

