import csv

class DataFile:
    def __init__(self, filename, columns=None):
        self.filename = filename
        self.writer = None
        self.columns = columns  # Save columns here for easy access

    def set_columns(self, col_list):
        self.columns = col_list

    def write_header(self):
        with open(self.filename, "w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()
        # Save fieldnames for later filtering
        self.fieldnames = self.columns

    def write_line(self, **kwargs):
        # Save fieldnames for later filtering
        if not hasattr(self, 'fieldnames'):
            self.fieldnames = self.columns

        with open(self.filename, "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writerow(kwargs)
